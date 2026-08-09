from sklearn.metrics.pairwise import cosine_similarity


class VectorStore:
    
    def __init__(self, chunks, vectorizer, embeddings):
        
        self.chunks = chunks
        self.vectorizer = vectorizer
        self.embeddings = embeddings
        
           
    def search(self, query:str, top_k:int = 1):
        
        print("\nchunks:\n")
        print(self.chunks)
        
        print("\nembeddings:\n")
        print(self.embeddings.shape)
        
        query_embedding = self.vectorizer.transform([query])
        print("\nquery_embedding:\n")
        print(query_embedding.shape)
                
        scores = cosine_similarity(
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
    