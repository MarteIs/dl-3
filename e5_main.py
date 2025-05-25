from metrics import recall_at_k, mrr
from retrieval import load_data, TransformerRetreiver
import torch

# Pretrained
# Recall@1: 0.8449
# Recall@3: 0.9387
# Recall@10: 0.9716
# MRR: 0.8957


if __name__ == "__main__":
    test = load_data(test=True)
    retriever = TransformerRetreiver()

    predict = retriever.retrieve(test)

    target = torch.arange(len(test))

    print(f"Recall@1: {recall_at_k(target, predict, k=1):.4f}",
          f"Recall@3: {recall_at_k(target, predict, k=3):.4f}",
          f"Recall@10: {recall_at_k(target, predict, k=10):.4f}",
          f"MRR: {mrr(target, predict):.4f}", sep="\n")
