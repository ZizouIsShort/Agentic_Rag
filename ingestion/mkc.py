from ingestion.loader import load_pdf, load_hface
from ingestion.chunking import chunk_documents

def load_mkc():
    pdf_chunk = chunk_documents(load_pdf())
    return pdf_chunk

def load_kms():
    kms_chunk = chunk_documents(load_hface())
    return kms_chunk