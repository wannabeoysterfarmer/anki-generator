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

# ---- Config / Keys ----
# Put your key in Streamlit Secrets (recommended): Settings -> Secrets -> add OPENAI_API_KEY
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
A: A detailed and comprehensive, scientifically-backed explanation for effective studying.

Each card should test an important concept, mechanism, or relationship. No cloze deletions. Avoid trivia. Be educational and accurate.

Slide:
\"\"\"{slide_text}\"\"\"
"""
    for _ in range(retries):
        try:
            res = client.chat.completions.create(
                # Use a fast, cheap model; change if you have access to others
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
        except Exception as e:
            # Backoff a little and try again
            time.sleep(3)
    return []

def extract_slides(pdf_path: str, image_dir: str):
    """Return list of (slide_text, slide_image_path)."""
    doc = fitz.open(pdf_path)
    slides = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        img_path = os.path.join(image_dir, f"slide_{i+1}.png")
        pix = page.get_pixmap(dpi=150)
        pix.save(img_path)
        slides.append((text, img_path))
    return slides

def build_anki_deck(cards, deck_name: str) -> str:
    """cards: list[(q, a, image_path)] -> writes deck, returns file path."""
    model_id = int(hashlib.md5("basic_qa_model".encode()).hexdigest(), 16) % (10**10)
    model = Model(
        model_id,
        "Basic QA Model",
        fields=[{"name": "Front"}, {"name": "Back"}, {"name": "Extra"}],
        templates=[{
            "name": "Card 1",
            "qfmt": "{{Front}}",
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
            text_for_model = text.strip() if text else "No text on this slide; infer the key concept from the image filename only."
            qa_list = generate_qa_cards(text_for_model, max_cards=max_cards_per_slide)
            for q, a in qa_list:
                anki_cards.append((q, a, image_path))

        if not anki_cards:
            return "No cards were generated.", None

        deck_name = f"{Path(pdf_path).stem} - Generated Anki Deck"
        apkg_path = build_anki_deck(anki_cards, deck_name)

        # Read bytes so Streamlit can offer a download button
        with open(apkg_path, "rb") as f:
            apkg_bytes = f.read()

    return "Deck created successfully!", (deck_name + ".apkg", apkg_bytes)

# ---- UI ----
st.set_page_config(page_title="PDF → Anki Deck", page_icon="📚")
st.title("📚 PDF → Anki Deck Generator")

uploaded_pdf = st.file_uploader("Upload your lecture PDF", type=["pdf"])
max_cards = st.slider("Cards per slide", min_value=1, max_value=5, value=1, step=1)

if st.button("Generate Deck"):
    with st.spinner("Processing..."):
        status, result = process_pdf_and_generate_deck(uploaded_pdf, max_cards)
    st.write(status)
    if result:
        fname, data = result
        st.download_button("Download Anki Deck (.apkg)", data=data, file_name=fname, mime="application/octet-stream")
