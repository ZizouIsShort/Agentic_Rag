import os
from google import genai
from ingestion.mkc import load_mkc, load_kms

GOOGLE_API_KEY = os.getenv("ZizouRAG")
client = genai.Client(api_key=GOOGLE_API_KEY)


def load_embeddings():
    all_chunks = load_mkc() + load_kms()
    embedded_chunks = []

    for chunk in all_chunks:
        text = chunk.get("text")
        if not text:
            continue

        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text
        )

        embedded_chunks.append({
            "text": text,
            "metadata": chunk["metadata"],
            "embedding": result.embeddings[0].values
        })

    return embedded_chunks
