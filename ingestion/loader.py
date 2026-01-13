from pathlib import Path
from pypdf import PdfReader
from typing import List, Dict

DATA_DIR = Path("data/raw")

def load_pdf() -> List[Dict]:
    documents = []
    for pdf in DATA_DIR.glob("*.pdf"):
        reader = PdfReader(pdf)

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()

            if not text or not text.strip():
                continue

            documents.append({
                "text": text.strip(),
                "metadata": {
                    "source_type": "pdf",
                    "filename": pdf.name,
                    "page_number": page_number
                }
            })
    return documents