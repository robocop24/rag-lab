import numpy as np


class VectorStore:
    
    def __init__(self, chunks, vectorizer, embeddings):
        
        self.chunks = chunks
        self.vectorizer = vectorizer
        self.embeddings = embeddings
        
    def cosine_similarity(self, v1, v2):
        return np.dot(v1, v2.T) / (np.linalg.norm(v1) * np.linalg.norm(v2, axis=1))
        
    def search(self, query:str, top_k:int = 1):
        
        print("\nchunks:\n")
        print(self.chunks)
        
        print("\nembeddings:\n")
        print(self.embeddings.shape)
        
        query_embedding = self.vectorizer.encode([query])
        print("\nquery_embedding:\n")
        print(query_embedding.shape)
                
        scores = self.cosine_similarity(
            query_embedding, self.embeddings
        ).flatten()
        print("\nscores:\n")
        print(scores)
        
        ranked = sorted(
            zip(self.chunks, scores),
            key=lambda x: x[1],
            reverse=True
        )
        print("\nranked:\n")
        print(ranked)
        
        return ranked[:top_k]
    