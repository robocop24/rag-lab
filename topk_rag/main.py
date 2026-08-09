import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from chunker import load_and_chunk
from embeddings import EmbeddingModel
from faiss_store import FaissStore
from retriever import Retriever

from llm.llm import LLM

chunks = load_and_chunk("documents/portal.txt")

embedding_model = EmbeddingModel()

chunk_embeddings = embedding_model.embed_documents(chunks)

vector_store = FaissStore(chunks, chunk_embeddings)

retriever = Retriever(vector_store=vector_store)

llm = LLM()

query = input("Question: ")

query_embedding = embedding_model.embed_query(query)

retrieved_chunks = retriever.retrieve(query_embedding, tok_k=3)

context = "\n\n".join(r["chunk"] for r in retrieved_chunks)
    
prompt = f"""
    Answere the question using the provided context only.
    
    Context: {context}
    
    Question: {query}
    """
    
answere = llm.generate(prompt)

print("\nRetrieved Context:\n")
print(context)

print("\nAnwsere: \n")
print(answere)