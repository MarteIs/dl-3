import torch
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TFIDFRetriever:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.answer_tfidf = None

    def fit(self, answers: List[str]):
        self.answer_tfidf = self.vectorizer.fit_transform(answers)

    def retrieve(self, answers: List[str], queries: List[str]):
        answers_encoded = self.vectorizer.transform(answers)
        queries_encoded = self.vectorizer.transform(queries)

        sims = torch.Tensor(cosine_similarity(queries_encoded, answers_encoded))
        indices = sims.sort(dim=1, descending=True).indices

        return indices