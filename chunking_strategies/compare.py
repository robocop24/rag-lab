from pathlib import Path

from fixed_chunker import chunk_text as fixed_chunk
from overlap_chunker import chunk_text as overlap_chunk
from paragraph_chunker import chunk_text as paragraph_chunk
from semantic_chunker import chunk_text as semantic_chunk

FILE_PATH = str(Path(__file__).parent.parent / "documents" / "portal.txt")

def print_chunks(title, chunks):
    
    print(f"\n{'=' * 50}")
    
    print(title)
    
    print(f"\n{'=' * 50}")
    
    print(f"Chunk Count: {len(chunks)}\n")
    
    for idx, chunk in enumerate(chunks, start=1):
        
        print(f"Chunk {idx}")
        
        print("-"* 20)
        
        print(chunk)
        
        print()
        
def main():
    
    paragraph_chunks = paragraph_chunk(FILE_PATH);
    
    fixed_chunks = fixed_chunk(FILE_PATH, chunk_size=50);
    
    overlap_chunks = overlap_chunk(
        FILE_PATH, 
        chunk_size=50,
        overlap=15
        );
    
    semantic_chunks = semantic_chunk(FILE_PATH)
    
    print("Paragraph chunks: ", paragraph_chunks)
    
    print("Fixed chunks: ", fixed_chunks)
    
    print("Overlap chunks: ", overlap_chunks)
    
    print_chunks("Semantic chunks: ", semantic_chunks)
    
if __name__ == "__main__":
    main()