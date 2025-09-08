import os
import time
import tempfile
import random
import hashlib
from pathlib import Path

import streamlit as st
import fitz  # PyMuPDF
from genanki import Note, Model, Deck, Package
from openai import OpenAI

# add at top with other imports
import base64
from functools import lru_cache

@st.cache_data(show_spinner=False)
def _pdf_byteskey(pdf_bytes: bytes) -> str:
    # small key so cache invalidates when pdf changes
    import hashlib
    return hashlib.sha1(pdf_bytes).hexdigest()

@st.cache_data(show_spinner=False)
def make_thumbnails_cached(pdf_bytes: bytes, dpi: int = 90) -> list[tuple[int, bytes]]:
    """
    Return [(page_num, png_bytes)] for ALL pages.
    Cached by the content of the uploaded PDF.
    """
    key = _pdf_byteskey(pdf_bytes)  # used implicitly by cache
    import io, fitz
    thumbs = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=dpi)
        buf = io.BytesIO(pix.tobytes("png"))
        thumbs.append((i, buf.getvalue()))
    return thumbs

def init_selection(n_pages: int):
    sel = st.session_state.get("selected_pages_set")
    if sel is None:
        st.session_state.selected_pages_set = set(range(1, n_pages + 1))  # default: all
        
def _img_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# --- Optional OCR deps (safe import) ---
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# ---- Config / Keys ----
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("No OpenAI API key found. Add OPENAI_API_KEY to Streamlit secrets or env.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ---- Core logic ----
def generate_qa_cards(slide_text: str, max_cards: int = 1, retries: int = 3):
    prompt = f"""
You are an expert tutor generating flashcards from lecture slides.

Please analyze the following slide text and return up to {max_cards} high-quality Anki-style flashcards in this format:

Q: What is the question?
A: An accurate, tightened, scientifically-backed explanation for effective studying.

Each card should test an important concept, mechanism, or relationship. No cloze deletions. Avoid trivia. Be educational and accurate.

Slide:
\"\"\"{slide_text}\"\"\"
"""
    for _ in range(retries):
        try:
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            text = (res.choices[0].message.content or "").strip()
            lines = [line.strip() for line in text.split("\n") if line.strip()]

            qa_pairs, current_q, current_a = [], None, None
            for line in lines:
                if line.startswith("Q:"):
                    if current_q and current_a:
                        qa_pairs.append((current_q, current_a))
                    current_q = line[2:].strip()
                    current_a = ""
                elif line.startswith("A:"):
                    current_a = line[2:].strip()
                elif current_a is not None:
                    current_a += " " + line.strip()
            if current_q and current_a:
                qa_pairs.append((current_q, current_a))
            return qa_pairs[:max_cards]
        except Exception:
            time.sleep(3)
    return []

def extract_slides(pdf_path: str, image_dir: str, selected_pages: list[int] | None = None):
    """Return list of (slide_text, slide_image_path, page_num)."""
    doc = fitz.open(pdf_path)
    slides = []
    page_indices = [(i+1) for i in range(len(doc))] if not selected_pages else selected_pages

    for pnum in page_indices:
        page = doc[pnum - 1]
        text = (page.get_text() or "").strip()
        img_path = os.path.join(image_dir, f"slide_{pnum}.png")
        page.get_pixmap(dpi=150).save(img_path)
        slides.append((text, img_path, pnum))
    return slides

def extract_text_with_ocr(image_path: str) -> str:
    if not OCR_AVAILABLE:
        return ""
    try:
        with Image.open(image_path) as im:
            im = im.convert("L")
            w, h = im.size
            if max(w, h) < 1200:
                scale = 1200 / max(w, h)
                im = im.resize((int(w*scale), int(h*scale)))
        text = pytesseract.image_to_string(im)
        return (text or "").strip()
    except Exception:
        return ""

def make_thumbnails_for_selection(pdf_path: str, tmpdir: str) -> list[tuple[int, str]]:
    doc = fitz.open(pdf_path)
    thumbs = []
    for i, page in enumerate(doc, start=1):
        img_path = os.path.join(tmpdir, f"thumb_{i}.png")
        page.get_pixmap(dpi=100).save(img_path)
        thumbs.append((i, img_path))
    return thumbs

def build_anki_deck(cards, deck_name: str) -> str:
    model_id = int(hashlib.md5("basic_qa_model".encode()).hexdigest(), 16) % (10**10)
    model = Model(
        model_id,
        "Basic QA Model",
        fields=[{"name": "Front"}, {"name": "Back"}, {"name": "Extra"}],
        templates=[{
            "name": "Card 1",
            "qfmt": "{{Front}}",  # no image on question side
            "afmt": "{{Front}}<hr id='answer'>{{Back}}<br>{{Extra}}",  # image on back via Extra
        }],
    )

    output_file = f"{deck_name}.apkg"
    deck_id = random.randrange(1 << 30, 1 << 31)
    deck = Deck(deck_id, deck_name)
    media_files = []

    for (q, a, img_path) in cards:
        extra_html = ""
        if img_path:
            extra_html = f"<br><img src='{Path(img_path).name}'>"
            media_files.append(img_path)
        note = Note(
            model=model,
            fields=[q, a, extra_html],
            tags=["autogen"],
            guid=str(hash((q, a))),
        )
        deck.add_note(note)

    pkg = Package(deck)
    pkg.media_files = media_files
    pkg.write_to_file(output_file)
    return output_file

def process_pdf_and_generate_deck(
    uploaded_file,
    max_cards_per_slide: int = 1,
    selected_pages: list[int] | None = None,
):
    if uploaded_file is None:
        return "Please upload a PDF.", None

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save uploaded PDF
        pdf_path = os.path.join(tmpdir, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Render slides (maybe only selected pages)
        slides = extract_slides(pdf_path, tmpdir, selected_pages=selected_pages)

        # Generate cards
        anki_cards = []
        for _, (text, image_path, _pnum) in enumerate(slides, start=1):
            if not text.strip():
                ocr_text = extract_text_with_ocr(image_path)
                text_for_model = (
                    ocr_text if ocr_text
                    else "No readable text; infer a sensible question from typical med-school slide content (title, axes, labels may be unreadable)."
                )
            else:
                text_for_model = text.strip()

            qa_list = generate_qa_cards(text_for_model, max_cards=max_cards_per_slide)
            for q, a in qa_list:
                anki_cards.append((q, a, image_path))

        if not anki_cards:
            return "No cards were generated.", None

        deck_name = f"{Path(pdf_path).stem} - Generated Anki Deck"
        apkg_path = build_anki_deck(anki_cards, deck_name)

        with open(apkg_path, "rb") as f:
            apkg_bytes = f.read()

    return "Deck created successfully!", (deck_name + ".apkg", apkg_bytes)

# ---- UI ----
# ---- UI ----
st.set_page_config(page_title="PDF → Anki Deck", page_icon="📚")
st.title("📚 PDF → Anki Deck Generator")

if not OCR_AVAILABLE:
    st.info(
        "OCR fallback not available. To enable OCR for image-only slides, "
        "add `pytesseract`, `Pillow` to requirements and install the `tesseract-ocr` package."
    )

uploaded_pdf = st.file_uploader("Upload your lecture PDF", type=["pdf"])
max_cards = st.slider("Cards per slide", min_value=1, max_value=5, value=1, step=1)

selected_pages = None
if uploaded_pdf is not None:
    pdf_bytes = uploaded_pdf.getvalue()

    # 1) Cached thumbnails (MUCH faster on rerun)
    thumbs = make_thumbnails_cached(pdf_bytes, dpi=90)
    n_pages = len(thumbs)
    init_selection(n_pages)

    st.subheader("Select slides to include")
    st.caption("Use the paginator. Click checkboxes and press **Apply selection**.")

    # 2) Pagination + thumbs/row (bigger thumbs → fewer per row)
    colA, colB, colC = st.columns([2,2,3])
    with colA:
        thumbs_per_row = st.selectbox("Thumbnails per row", [2, 3, 4], index=1)
    with colB:
        page_size = st.selectbox("Slides per page", [8, 12, 15, 20, 30], index=2)
    with colC:
        # compute total pages
        import math
        total_pages = math.ceil(n_pages / page_size)
        page_idx = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)

    start = (page_idx - 1) * page_size
    end = min(start + page_size, n_pages)
    page_slice = thumbs[start:end]

    # 3) Bulk controls for the *current* page only
    c1, c2, _ = st.columns([1,1,6])
    with c1:
        if st.button("Uncheck All (page)"):
            for pnum, _ in page_slice:
                st.session_state.selected_pages_set.discard(pnum)
    with c2:
        if st.button("Check All (page)"):
            for pnum, _ in page_slice:
                st.session_state.selected_pages_set.add(pnum)

    # 4) Selection grid inside a form → no re-render until submit
    with st.form(key=f"selection_form_page_{page_idx}"):
        # grid rows
        for row_start in range(0, len(page_slice), thumbs_per_row):
            row_items = page_slice[row_start: row_start + thumbs_per_row]
            cols = st.columns(len(row_items))
            for col, (pnum, png_bytes) in zip(cols, row_items):
                with col:
                    # show image bytes directly (faster than base64)
                    st.image(png_bytes, caption=f"Slide {pnum}", use_container_width=True)
                    key = f"sel_{pnum}"
                    default_checked = (pnum in st.session_state.selected_pages_set)
                    checked = st.checkbox("Select", value=default_checked, key=key)
                    # we **don’t** modify the set yet; do it on submit below
        apply_now = st.form_submit_button("Apply selection")

        if apply_now:
            # Read all checkboxes shown on this page and write back to the set once
            for (pnum, _png) in page_slice:
                key = f"sel_{pnum}"
                if st.session_state.get(key, False):
                    st.session_state.selected_pages_set.add(pnum)
                else:
                    st.session_state.selected_pages_set.discard(pnum)

    # 5) Summary + ability to select all / none for the whole doc (optional)
    s1, s2, s3 = st.columns([2,2,6])
    with s1:
        if st.button("Select NONE (all pages)"):
            st.session_state.selected_pages_set = set()
    with s2:
        if st.button("Select ALL (all pages)"):
            st.session_state.selected_pages_set = set(range(1, n_pages + 1))

    st.write(f"Selected: **{len(st.session_state.selected_pages_set)} / {n_pages}**")

    # Build selected_pages list for processing
    selected_pages = sorted(list(st.session_state.selected_pages_set))
    if len(selected_pages) == 0:
        st.info("No slides selected — generating from **all** slides.")
        selected_pages = None

# Generate
if st.button("Generate Deck"):
    with st.spinner("Processing..."):
        status, result = process_pdf_and_generate_deck(
            uploaded_file=uploaded_pdf,
            max_cards_per_slide=max_cards,
            selected_pages=selected_pages if uploaded_pdf else None,
        )
    st.write(status)
    if result:
        fname, data = result
        st.download_button(
            "Download Anki Deck (.apkg)",
            data=data,
            file_name=fname,
            mime="application/octet-stream",
        )
