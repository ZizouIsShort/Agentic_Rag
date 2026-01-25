from typing import List, Dict

def chunk_documents(
    documents: List[Dict],
    chunk_size: int = 600,
    overlap: int = 100
) -> List[Dict]:
    chunks = []

    for doc in documents:
        text = doc["text"]
        metadata = doc["metadata"]

        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            if chunk_text.strip():
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = chunk_index

                chunks.append({
                    "text": chunk_text.strip(),
                    "metadata": chunk_metadata
                })

            start += chunk_size - overlap
            chunk_index += 1

    return chunks
