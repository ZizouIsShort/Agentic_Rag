from pathlib import Path
from pypdf import PdfReader
from typing import List, Dict
from datasets import load_dataset


DATA_DIR = Path("data/raw")


def load_hface(limit: int = 1) -> List[Dict]:
    ds = load_dataset("SustcZhangYX/ChatEnv")
    dataset_split = ds["train"]

    hfacedoc = []
    count = 0

    for idx, row in enumerate(dataset_split):
        question = row.get("instruction")
        answer = row.get("output")

        if not question or not answer:
            continue

        question = question.strip()
        answer = answer.strip()

        if not question or not answer:
            continue

        hfacedoc.append({
            "text": f"Question:\n{question}\n\nAnswer:\n{answer}",
            "metadata": {
                "source_type": "huggingface",
                "dataset": "SustcZhangYX/ChatEnv",
                "split": "train",
                "row_id": idx
            }
        })

        count += 1
        if count >= limit:
            break

    return hfacedoc


def load_pdf(limit: int = 10) -> List[Dict]:
    documents = []
    for pdf in DATA_DIR.glob("*.pdf"):
        reader = PdfReader(pdf)

        for page_number, page in enumerate(reader.pages, start=1):
            if len(documents) >= limit:
                return documents

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
