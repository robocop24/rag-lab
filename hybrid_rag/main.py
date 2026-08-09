from pathlib import Path

from chunker import chunk_text
from embeddings import EmbeddingModel
from hybrid_search import HybridSearch

DOCUMENTS_DIR = Path(__file__).parent.parent / "documents"

chunks = chunk_text(str(DOCUMENTS_DIR / "portal.txt"))

embedding_model = EmbeddingModel()

chunk_embeddings = embedding_model.embed_documents(chunks)

hybrid_search = HybridSearch(chunks, chunk_embeddings)

query = input("Question: ")

query_embedding = embedding_model.embed_query(query)

results = hybrid_search.search(query, query_embedding, top_k=3)

print("\nResults:\n")

for result in results:
    
    print("*" * 50)
    
    print("Chunk: ", f"{result['chunk']}")
    
    print("Semantic Score: ", f"{result['semantic_score']:.4f}")
    
    print("Keyword Score: ", f"{result['keyword_score']}")
    
    print("Hybrid Score: ", f"{result['hybrid_score']:.4f}")