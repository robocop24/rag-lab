import faiss
import numpy as np


class FaissStore:
    
    def __init__(self, chunks, embeddings):
        
        self.chunks = chunks
        
        self.embeddings = (
            np.array(embeddings).astype("float32")
        )
        
        embeddings_dimension = self.embeddings.shape[1]
        
        self.index = faiss.IndexFlatL2(embeddings_dimension)
        
        self.index.add(self.embeddings)
        
        
    def search(self, query_embedding, tok_k=3):
        
        query_embedding = np.array(
            [query_embedding]).astype("float32")
        
        distances, indices = (
            self.index.search(query_embedding, tok_k)
        )
        print("\nChunks embedding index: \n")
        print(self.index)
        print("\ndistances, indices : \n")
        print(distances, indices)
        
        return (distances, indices)