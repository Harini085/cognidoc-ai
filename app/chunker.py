
import re


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[dict]:
    """
    Create paragraph-aware chunks from PDF pages.

    Preserves source filenames and page numbers.
    Prefers paragraph and sentence boundaries where possible.
    Chunk sizes are measured in characters.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    chunks = []

    def split_long_text(text: str) -> list[str]:
        """Split oversized paragraphs at sentence or word boundaries."""
        parts = []
        remaining = text.strip()

        while len(remaining) > chunk_size:
            limit = chunk_size

            # Prefer a sentence boundary near the chunk limit.
            sentence_matches = list(
                re.finditer(r'[.!?](?:\s+|$)', remaining[:limit])
            )

            boundary = 0
            if sentence_matches:
                candidate = sentence_matches[-1].end()
                if candidate >= chunk_size // 2:
                    boundary = candidate

            # If no suitable sentence boundary exists, use a word boundary.
            if boundary == 0:
                boundary = remaining.rfind(" ", 0, limit)

            if boundary < chunk_size // 2:
                boundary = limit

            part = remaining[:boundary].strip()
            if part:
                parts.append(part)

            # Retain some context between consecutive pieces.
            next_start = max(0, boundary - overlap)

            # Avoid repeatedly selecting the same boundary.
            if next_start >= len(remaining):
                break

            remaining = remaining[next_start:].strip()

            # Ensure progress if overlap would otherwise stall the loop.
            if len(remaining) >= len(text) and not parts:
                remaining = remaining[chunk_size:].strip()

        if remaining:
            parts.append(remaining)

        return parts

    for page_data in pages:
        raw_text = page_data.get("text", "")

        # Preserve paragraph boundaries instead of flattening all whitespace.
        paragraphs = [
            re.sub(r"\s+", " ", paragraph).strip()
            for paragraph in re.split(r"\n\s*\n", raw_text)
        ]
        paragraphs = [p for p in paragraphs if p]

        # PDFs may not provide blank lines between paragraphs.
        if len(paragraphs) <= 1:
            paragraphs = [
                re.sub(r"\s+", " ", raw_text).strip()
            ]
            paragraphs = [p for p in paragraphs if p]

        pieces = []

        for paragraph in paragraphs:
            if len(paragraph) > chunk_size:
                pieces.extend(split_long_text(paragraph))
            else:
                pieces.append(paragraph)

        current = ""

        for piece in pieces:
            candidate = f"{current}\n\n{piece}".strip() if current else piece

            if len(candidate) <= chunk_size:
                current = candidate
                continue

            if current:
                chunks.append({
                    "chunk_id": len(chunks),
                    "source": page_data["source"],
                    "page": page_data["page"],
                    "text": current,
                })

            # Keep a short trailing context from the previous chunk.
            if current and overlap:
                context = current[-overlap:].strip()
                current = f"{context}\n\n{piece}".strip()
            else:
                current = piece

            # A paragraph plus overlap may exceed the target slightly.
            # Split it safely if needed.
            if len(current) > chunk_size:
                subparts = split_long_text(current)
                current = ""

                for subpart in subparts[:-1]:
                    chunks.append({
                        "chunk_id": len(chunks),
                        "source": page_data["source"],
                        "page": page_data["page"],
                        "text": subpart,
                    })

                if subparts:
                    current = subparts[-1]

        if current:
            chunks.append({
                "chunk_id": len(chunks),
                "source": page_data["source"],
                "page": page_data["page"],
                "text": current,
            })

    return chunks
