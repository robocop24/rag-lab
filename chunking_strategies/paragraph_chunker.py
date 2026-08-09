from pathlib import Path


def chunk_text(file_path: str) -> list:
    """Split document by paragraphs."""
    content = Path(file_path).read_text(encoding="utf-8")

    chunks = [
        chunk.strip()
        for chunk in content.split("\n\n")
        if chunk.strip()
    ]

    return chunks