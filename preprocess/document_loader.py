
# chunking the whole pdf then maping with the page number so that we can solve the non-uniform chunk length distribution

import os
import re
import ftfy
import unicodedata

def preprocess_text(text):
    # Fix broken text like don‚Äôt -> don't
    text = ftfy.fix_text(text)
    # Normalize unicode (e.g., é → e, ' → ')
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    # Keep only allowed characters: letters, numbers, bullets, punctuation
    text = re.sub(r"[^a-zA-Z0-9\s.,!?•*-]", "", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_pdf_with_pages(file_path):
    """
    Loads PDF and returns a list of dicts: [{ "page": page_number, "text": preprocessed_text }]
    """
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        raise ImportError("PyPDF2 is required for PDF support. Install with: pip install PyPDF2")
    reader = PdfReader(file_path)
    results = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        page_text = preprocess_text(page_text)
        results.append({"page": i + 1, "text": page_text})
    return results

def load_text(file_path):
    """
    Loads and extracts text from .txt, .md, .docx.
    For PDFs, use load_pdf_with_pages instead!
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".txt", ".md"]:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return preprocess_text(text)

    elif ext == ".pdf":
        # Use the new function for page-wise extraction!
        return load_pdf_with_pages(file_path)

    elif ext == ".docx":
        try:
            import docx
        except ImportError:
            raise ImportError("python-docx is required for DOCX support. Install with: pip install python-docx")
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return preprocess_text(text)

    else:
        raise ValueError(f"Unsupported file extension: {ext}. Supported: .txt, .md, .pdf, .docx")