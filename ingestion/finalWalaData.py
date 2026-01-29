from ingestion.embedding import load_embeddings
from pinecone import Pinecone
import os
from pinecone.grpc import PineconeGRPC as Pinecone

def mkbfinal_data():
    final_data = load_embeddings()
    return final_data
