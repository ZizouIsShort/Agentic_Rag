import math
from typing import List



def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = math.sqrt(sum(a * a for a in vec1))
    norm_b = math.sqrt(sum(b * b for b in vec2))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)

def retrieve_top_chunks(
    query_embedding: List[float],
    embedded_chunks: list,
    top_k: int = 5,
    min_score: float = 0.2
):
    scored_chunks = []

    for chunk in embedded_chunks:
        score = cosine_similarity(
            query_embedding,
            chunk["embedding"]
        )

        if score >= min_score:
            scored_chunks.append({
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "score": score
            })

    scored_chunks.sort(key=lambda x: x["score"], reverse=True)

    return scored_chunks[:top_k]