def build_context(chunks: list) -> str:
    context_parts = []

    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"Chunk {i}:\n{chunk['text']}"
        )

    return "\n\n---\n\n".join(context_parts)
