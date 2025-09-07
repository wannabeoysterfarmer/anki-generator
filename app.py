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

# --- Optional OCR deps (safe import) ---
# If pytesseract/Pillow or tesseract binary aren't present, we keep running without OCR.
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
    """Call OpenAI to make up to N Q/A pairs for a slide; returns list[(q, a)]."""
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
                model="gpt-4o-mini",  # change if you prefer a different model
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

def extract_slides(pdf_path: str, image_dir: str):
    """Return list of (slide_text, slide_image_path)."""
    doc = fitz.open(pdf_path)
    slides = []
    for i, page in enumerate(doc):
        text = (page.get_text() or "").strip()
        img_path = os.path.join(image_dir, f"slide_{i+1}.png")
        pix = page.get_pixmap(dpi=150)
        pix.save(img_path)
        slides.append((text, img_path))
    return slides

def extract_text_with_ocr(image_path: str) -> str:
    """OCR the rendered slide image to recover text when PDF text is empty."""
    if not OCR_AVAILABLE:
        return ""
    try:
        # Light preprocessing helps OCR a bit
        with Image.open(image_path) as im:
            im = im.convert("L")  # grayscale
            # Small upscale for tiny text
            w, h = im.size
            if max(w, h) < 1200:
                scale = 1200 / max(w, h)
                im = im.resize((int(w*scale), int(h*scale)))
        text = pytesseract.image_to_string(im)
        return (text or "").strip()
    except Exception:
        return ""

def build_anki_deck(cards, deck_name: str) -> str:
    """cards: list[(q, a, image_path)] -> writes deck, returns file path."""
    model_id = int(hashlib.md5("basic_qa_model".encode()).hexdigest(), 16) % (10**10)
    model = Model(
        model_id,
        "Basic QA Model",
        fields=[{"name": "Front"}, {"name": "Back"}, {"name": "Extra"}],
        # Put image ONLY on the answer (Extra is shown on back below Back)
        templates=[{
            "name": "Card 1",
            "qfmt": "{{Front}}",  # no image on question side
            "afmt": "{{Front}}<hr id='answer'>{{Back}}<br>{{Extra}}",
        }],
    )

    output_file = f"{deck_name}.apkg"
    deck_id = random.randrange(1 << 30, 1 << 31)
    deck = Deck(deck_id, deck_name)
    media_files = []

    for (q, a, img_path) in cards:
        extra_html = ""
        if img_path:
            # Reference by basename; include full path in media_files
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

def process_pdf_and_generate_deck(uploaded_file, max_cards_per_slide: int = 1):
    """Main pipeline: save PDF, render slides, call OpenAI, build deck, return bytes."""
    if uploaded_file is None:
        return "Please upload a PDF.", None

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save uploaded PDF
        pdf_path = os.path.join(tmpdir, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Render slides to images + extract text
        slides = extract_slides(pdf_path, tmpdir)

        # Generate cards
        anki_cards = []
        for i, (text, image_path) in enumerate(slides, start=1):
            if not text.strip():
                # OCR fallback when the PDF page doesn't expose selectable text
                ocr_text = extract_text_with_ocr(image_path)
                text_for_model = ocr_text if ocr_text else "No readable text; infer a sensible question from typical med-school slide content (title, axes, labels are unreadable)."
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
st.set_page_config(page_title="PDF → Anki Deck", page_icon="📚")
st.title("📚 PDF → Anki Deck Generator")

if not OCR_AVAILABLE:
    st.info("OCR fallback not available. To enable OCR for image-only slides, add `pytesseract`, `Pillow` to requirements and `tesseract-ocr` to packages.")

uploaded_pdf = st.file_uploader("Upload your lecture PDF", type=["pdf"])
max_cards = st.slider("Cards per slide", min_value=1, max_value=5, value=1, step=1)

if st.button("Generate Deck"):
    with st.spinner("Processing..."):
        status, result = process_pdf_and_generate_deck(uploaded_pdf, max_cards)
    st.write(status)
    if result:
        fname, data = result
        st.download_button("Download Anki Deck (.apkg)", data=data, file_name=fname, mime="application/octet-stream")
