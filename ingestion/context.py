def build_context(chunks: list[dict]) -> str:
    context_parts = []

    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})

        source = meta.get("source_type", "unknown")
        filename = meta.get("filename", "")
        page = meta.get("page_number", "")
        dataset = meta.get("dataset", "")
        row_id = meta.get("row_id", "")

        source_line = f"Source: {source}"
        if filename:
            source_line += f" | File: {filename}"
        if page:
            source_line += f" | Page: {page}"
        if dataset:
            source_line += f" | Dataset: {dataset}"
        if row_id != "":
            source_line += f" | Row: {row_id}"

        context_parts.append(
            f"""Chunk {i}
{source_line}

{chunk['text']}"""
        )

    return "\n\n---\n\n".join(context_parts)
