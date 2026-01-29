from ingestion.embedding import load_embeddings
from pinecone import Pinecone
import os

def mkbfinal_data():
    final_data = load_embeddings()

    pc = Pinecone(api_key=os.getenv("ZizouKaRAG"))
    index = pc.Index("developer-quickstart-py")

    vectors = []

    for i, chunk in enumerate(final_data):
        vectors.append({
            "id": f"chunk-{i}",
            "values": chunk["embedding"],
            "metadata": {
                "text": chunk["text"],
                **chunk["metadata"]
            }
        })

    index.upsert(vectors=vectors)

    print(f"Upserted {len(vectors)} vectors into Pinecone")
