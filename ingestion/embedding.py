import os
from google import genai
from ingestion.mkc import load_mkc, load_kms

GOOGLE_API_KEY = os.getenv("ZizouRAG")
client = genai.Client(api_key=GOOGLE_API_KEY)


def load_embeddings():
    all_chunks = load_mkc() + load_kms()
    embedded_chunks = []

    texts = [c["text"] for c in all_chunks if c.get("text")]

    results = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts
    )

    for chunk, emb in zip(all_chunks, results.embeddings):
        embedded_chunks.append({
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "embedding": emb.values
        })

    return embedded_chunks
