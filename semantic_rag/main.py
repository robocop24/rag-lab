from chunker import load_and_chunk
from embeddings import generate_embeddings
from vector_store import VectorStore
from pathlib import Path

DOC_PATH = Path(__file__).parent.parent / "documents" / "portal.txt"

chunks = load_and_chunk(str(DOC_PATH))

vectorizer, embeddings = generate_embeddings(chunks)

store = VectorStore(chunks,vectorizer,embeddings)

query = input("Question: ")

results = store.search(query=query, top_k=2)
print("\nresults:\n")
print(results)

print("\nBest Match:\n")
for chunk, score in results:
    
    print(f"Score :{score: .4f}")
    
    print(chunk)