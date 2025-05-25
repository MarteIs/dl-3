import torch
from retrieval.model import EmbeddingModel

class HardNegativeMiner:
    def __init__(self, model: EmbeddingModel):
        self.model = model
    
    def find_negatives(self, queries, docs, top_k=5):
        with torch.no_grad():
            q_emb = self.model(queries)
            d_emb = self.model(docs)
            sim = q_emb @ d_emb.T
            _, indices = torch.topk(sim, k=top_k+1, dim=1)
        return indices[:, 1:]