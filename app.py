import os
import time
import tempfile
import random
import hashlib
from pathlib import Path
import base64
from streamlit_cookies_manager import EncryptedCookieManager
import streamlit as st
import uuid
import time
cookies = EncryptedCookieManager(
    prefix="decksmith_",
    password=st.secrets["cookie_password"]
)

# ✅ Retry loop here to wait for readiness
for _ in range(5):
    if cookies.ready():
        break
    time.sleep(0.2)
else:
    st.warning("⚠️ Cookies not ready — try refreshing.")
    st.stop()

try:
    if "user_id" not in cookies:
        user_id = str(uuid.uuid4())[:8]
        cookies["user_id"] = user_id
        cookies.save()
    else:
        user_id = cookies["user_id"]
except Exception as e:
    st.warning(f"⚠️ Cookie error: {e}")
    user_id = "unknown"
import fitz  # PyMuPDF
from genanki import Note, Model, Deck, Package
from openai import OpenAI
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
from streamlit.runtime.scriptrunner import get_script_run_ctx

# ---------------- Google Drive Data Storage and Analytics ----------------
scope = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

creds = Credentials.from_service_account_info(
    st.secrets["google_service_account"], scopes=scope
)

gc = gspread.authorize(creds)
sheet = gc.open("Decksmith Analytics").sheet1

# ---------------- Cache helpers ----------------
@st.cache_data(show_spinner=False)
def _pdf_byteskey(pdf_bytes: bytes) -> str:
    import hashlib
    return hashlib.sha1(pdf_bytes).hexdigest()

@st.cache_data(show_spinner=False)
def make_thumbnails_cached(pdf_bytes: bytes, dpi: int = 90):
    """
    Return [(page_num, png_bytes)] for ALL pages.
    Cached by the content of the uploaded PDF.
    """
    import io, fitz
    _ = _pdf_byteskey(pdf_bytes)  # tie cache to bytes
    thumbs = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=dpi)
        buf = io.BytesIO(pix.tobytes("png"))
        thumbs.append((i, buf.getvalue()))
    return thumbs

def init_selection(n_pages: int):
    if "selected_pages_set" not in st.session_state:
        st.session_state.selected_pages_set = set(range(1, n_pages + 1))  # default: all

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

def extract_slides(pdf_path: str, image_dir: str, selected_pages=None):
    """Return list of (slide_text, slide_image_path, page_num)."""
    doc = fitz.open(pdf_path)
    slides = []
    page_indices = list(range(1, len(doc) + 1)) if not selected_pages else selected_pages

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

# ---------------- Google Drive Data Storage and Analytics Logging Function ----------------
def log_deck_generation_to_sheet(sheet, uploaded_file, deck_name, num_cards):
    try:
        # Get user's IP address (unofficial API; works on Streamlit Cloud + local)
        #ctx = get_script_run_ctx()
        #ip_address = ctx.remote_ip if ctx and hasattr(ctx, "remote_ip") else "unknown"
       
        # Simulate a persistent user ID using Streamlit session or a UUID fallback
        # Use persistent user_id from cookie
        sheet.append_row([
            datetime.utcnow().isoformat(),  # Timestamp
            user_id,                        # Persistent user ID
            len(final_selected) if final_selected else "unknown",  # Slide count
            num_cards,                      # Total number of cards
            uploaded_file.name              # PDF file name
        ])
    except Exception as e:
        st.warning(f"⚠️ Failed to log analytics: {e}")

def process_pdf_and_generate_deck(
    uploaded_file,
    max_cards_per_slide: int = 1,
    selected_pages=None,
    final_selected=None,  
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
        
        # Log to analytics sheet
        log_deck_generation_to_sheet(sheet, uploaded_file, deck_name, len(anki_cards))

        with open(apkg_path, "rb") as f:
            apkg_bytes = f.read()

    return "Deck created successfully!", (deck_name + ".apkg", apkg_bytes)

# ---------------- UI ----------------
st.set_page_config(page_title="Decksmith     ", page_icon="⚒️")
st.title("Introducing: Decksmith ⚒️ ")
st.title("Your all in one PDF → Anki Deck Generator 🔥🔥🔥")
st.markdown("""
### How to Use Decksmith

1. Upload a lecture PDF.
2. Select which slides you want included.
3. Choose how many cards per slide.
4. Click **Generate Deck** to create your Anki file.
5. When finished, download the deck and **refresh the page** before uploading another PDF.

⚠️ Decksmith currently supports one upload at a time. Please refresh between sessions to reset the app.
""")

if not OCR_AVAILABLE:
    st.info(
        "OCR fallback not available. To enable OCR for image-only slides, "
        "add `pytesseract`, `Pillow` to requirements and install the `tesseract-ocr` package."
    )

uploaded_pdf = st.file_uploader("Upload your lecture PDF", type=["pdf"])

st.markdown("""
#### 🎯 Recommendation:
We suggest generating **only 1–2 cards per slide**.

This keeps your deck **focused**, avoids unnecessary repetition, and helps you retain key concepts more efficiently. You can always edit or expand the cards afterward to match your learning style.
""")

max_cards = st.slider("Cards per slide", min_value=1, max_value=5, value=1, step=1)

selected_pages = None
if uploaded_pdf is not None:
    # Cache thumbnails by *content* so UI is snappy on re-renders
    pdf_bytes = uploaded_pdf.getvalue()
    pdf_key = _pdf_byteskey(pdf_bytes)

    # Reset selection set when a new PDF arrives
    if st.session_state.get("pdf_key") != pdf_key:
        st.session_state.pdf_key = pdf_key
        thumbs = make_thumbnails_cached(pdf_bytes, dpi=100)  # a touch bigger previews
        n_pages = len(thumbs)
        st.session_state.selected_pages_set = set(range(1, n_pages + 1))
    else:
        thumbs = make_thumbnails_cached(pdf_bytes, dpi=100)
        n_pages = len(thumbs)
        init_selection(n_pages)

    st.subheader("Select slides to include")
    st.caption("Selections apply instantly; paging won’t lose your choices.")

    # Layout controls
    # colA, colB, colC = st.columns([2, 2, 3])
    # with colA:
    #     thumbs_per_row = st.selectbox("Thumbnails per row", [2, 3, 4], index=1)
    # with colB:
    #     page_size = st.selectbox("Slides per page", [8, 12, 15, 20, 30], index=2)
    # with colC:
    #    import math
    #     total_pages = math.ceil(n_pages / page_size) if page_size else 1
    #     page_idx = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    # 
    # start = (page_idx - 1) * page_size
    # end   = min(start + page_size, n_pages)
    # page_slice = thumbs[start:end]  # [(pnum, png_bytes), ...]
  
    # Layout controls
    thumbs_per_row = 3 #st.selectbox("Thumbnails per row", [2, 3, 4], index=1)

    page_slice = thumbs  # Show all thumbnails

    # Bulk controls (current page only)
    c1, c2, _ = st.columns([2, 2, 6])
    with c1:
        if st.button("Uncheck All (page)"):
            for pnum, _ in page_slice:
                st.session_state.selected_pages_set.discard(pnum)
    with c2:
        if st.button("Check All (page)"):
            for pnum, _ in page_slice:
                st.session_state.selected_pages_set.add(pnum)

    # Grid (instant-apply)
    for row_start in range(0, len(page_slice), thumbs_per_row):
        row_items = page_slice[row_start: row_start + thumbs_per_row]
        cols = st.columns(len(row_items))
        for col, (pnum, png_bytes) in zip(cols, row_items):
            with col:
                st.image(png_bytes, caption=f"Slide {pnum}", use_container_width=True)
                key = f"sel_{pnum}"
                checked = (pnum in st.session_state.selected_pages_set)
                new_val = st.checkbox("Select", value=checked, key=key)
                if new_val:
                    st.session_state.selected_pages_set.add(pnum)
                else:
                    st.session_state.selected_pages_set.discard(pnum)

    # Global bulk actions (optional)
    s1, s2, _ = st.columns([2, 2, 6])
    with s1:
        if st.button("Select NONE (all pages)"):
            st.session_state.selected_pages_set = set()
    with s2:
        if st.button("Select ALL (all pages)"):
            st.session_state.selected_pages_set = set(range(1, n_pages + 1))

    # Live summary + final list
    st.write(f"Selected: **{len(st.session_state.selected_pages_set)} / {n_pages}**")
    selected_pages = sorted(st.session_state.selected_pages_set)
    if not selected_pages:
        st.info("No slides selected — generating from **all** slides.")
        selected_pages = None  # downstream treats None as “all”

# ---- Build Deck section (always visible so the button never disappears) ----
st.divider()
st.subheader("Build Deck")

final_selected = sorted(st.session_state.selected_pages_set) if "selected_pages_set" in st.session_state else None
if final_selected is not None and len(final_selected) == 0:
    st.info("No slides selected — generating from **all** slides.")
    final_selected = None  # None => all pages downstream

if st.button("Generate Deck", key="btn_generate"):
    if uploaded_pdf is None:
        st.warning("Please upload a PDF first.")
    else:
        with st.spinner("Processing..."):
            status, result = process_pdf_and_generate_deck(
                uploaded_file=uploaded_pdf,
                max_cards_per_slide=max_cards,
                selected_pages=final_selected,
                final_selected=final_selected  
            )
        st.write(status)
        if result:
            fname, data = result
            st.download_button(
                "Download Anki Deck (.apkg)",
                data=data,
                file_name=fname,
                mime="application/octet-stream"
            )
