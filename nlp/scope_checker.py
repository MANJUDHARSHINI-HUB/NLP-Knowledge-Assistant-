from sentence_transformers import SentenceTransformer, util

SCOPE_EXAMPLES = [
    "What is natural language processing?",
    "What are the types of NLP?",
    "Explain tokenization in NLP",
    "What is stemming and lemmatization?",
    "What is TF-IDF?",
    "What are word embeddings?",
    "Explain Word2Vec and GloVe",
    "What is sentiment analysis?",
    "What is named entity recognition?",
    "How does machine translation work?",
    "What is a transformer model?",
    "Explain self-attention",
    "What is BERT?",
    "What is GPT?",
    "What are large language models?",
    "What is prompt engineering?",
    "What is an embedding?",
    "What is semantic search?",
    "What is a vector database?",
    "What is ChromaDB?",
    "What is RAG?",
    "How does retrieval augmented generation work?",
    "How does this NLP knowledge assistant work?",
]

class ScopeChecker:
    def __init__(self, model, threshold=0.38):
        self.model = model
        self.threshold = threshold
        self.scope_embeddings = model.encode(SCOPE_EXAMPLES, convert_to_tensor=True)

    def check(self, query):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, self.scope_embeddings)[0]
        best_score = float(scores.max().item())
        return best_score >= self.threshold, best_score
