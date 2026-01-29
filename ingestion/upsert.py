from ingestion.embedding import load_embeddings
from pinecone import Pinecone
import os
from dotenv import load_dotenv
load_dotenv()

myren = os.getenv("myren")
def run_upsert():
    pc = Pinecone(api_key=myren)
    index = pc.Index("killme")
    data = "rag-data"
    chunks = load_embeddings()

    vectors = []
    for chunk in chunks:
        meta = chunk["metadata"]

        vector_id = (
            f"{meta.get('source_type')}-"
            f"{meta.get('filename', meta.get('dataset'))}-"
            f"{meta.get('page_number', meta.get('row_id'))}-"
            f"{meta.get('chunk_index')}"
        )

        vectors.append({
            "id": vector_id,
            "values": chunk["embedding"],
            "metadata": {
                **meta,
                "text": chunk["text"]
            }
        })

    index.upsert(vectors, data)
    print(f"Upserted {len(vectors)} vectors")
