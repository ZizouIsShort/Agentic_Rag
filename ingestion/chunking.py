from typing import List, Dict

def chunk_documents(
    documents: List[Dict],
    chunk_size: int = 600,
    overlap: int = 100,
    min_chunk_length: int = 50
) -> List[Dict]:
    chunks = []

    for doc in documents:
        text = doc["text"]
        base_metadata = doc["metadata"]

        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if (
                not chunk_text
                or len(chunk_text) < min_chunk_length
                or chunk_text.isdigit()
            ):
                start += chunk_size - overlap
                chunk_index += 1
                continue

            chunk_metadata = base_metadata.copy()
            chunk_metadata["chunk_index"] = chunk_index

            chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })

            start += chunk_size - overlap
            chunk_index += 1

    return chunks
