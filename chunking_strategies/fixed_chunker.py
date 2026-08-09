from pathlib import Path


def chunk_text(file_path: str, chunk_size:int = 100) -> list:
    
    """Split document into fixed-size chunks."""
    content = Path(file_path).read_text(encoding="utf-8")
    
    chunks = []
    
    for i in range(0, len(content), chunk_size):
        
        chunks.append(
            content[i:i+chunk_size]
        )

    return chunks