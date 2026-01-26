from typing import List
from mathsss.mfkinmaths import retrieve_top_chunks
from ingestion.context import build_context


def retrieve_context_for_query(
    query_embedding: List[float],
    embedded_chunks: list
) -> str:
    top_chunks = retrieve_top_chunks(
        query_embedding=query_embedding,
        embedded_chunks=embedded_chunks,
        top_k=5,
        min_score=0.2
    )

    if not top_chunks:
        return ""

    return build_context(top_chunks)
