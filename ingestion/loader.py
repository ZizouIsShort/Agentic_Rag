from pathlib import Path
from pypdf import PdfReader
from typing import List, Dict
from datasets import load_dataset

DATA_DIR = Path("data/raw")


def load_hface() -> List[Dict]:
    ds = load_dataset("SustcZhangYX/ChatEnv")
    hfacedoc = []
    dataset_split = ds["train"]
    print(dataset_split[0])
    print(dataset_split[0].keys())
    row = dataset_split[0]
    row.keys()
    for idx, row in enumerate(dataset_split):
        text = row.get("instruction")

        if not text or not text.strip():
            continue

        hfacedoc.append({
            "text": text.strip(),
            "metadata": {
                "source_type": "huggingface",
                "dataset": "SustcZhangYX/ChatEnv",
                "split": "train",
                "row_id": idx
            }
        })
    return hfacedoc
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