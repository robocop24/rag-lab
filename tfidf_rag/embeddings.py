from sklearn.feature_extraction.text import TfidfVectorizer


def generate_embeddings(chunks:list[str]):
    
    vectorizer = TfidfVectorizer()
    
    embeddings = vectorizer.fit_transform(chunks)
    
    return vectorizer, embeddings