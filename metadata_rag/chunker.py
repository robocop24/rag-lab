from pathlib import Path


def chunk_text(
    file_path: str, 
    chunk_size:int = 100,
    overlap:int = 20) -> list:
    
    """Split document into overlapping chunks."""
    content = Path(file_path).read_text(encoding="utf-8")
    
    chunks = []
    start  = 0
    
    while start < len(content):
        
        end = start + chunk_size
        
        chunks.append(content[start:end])
        
        start += chunk_size - overlap

    return chunks