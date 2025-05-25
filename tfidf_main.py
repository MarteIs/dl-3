from datasets import load_dataset
from retrieval import TFIDFRetriever
from metrics import recall_at_k, mrr
import torch

if __name__ == "__main__":
    tf_idf = TFIDFRetriever()

    data = load_dataset("sentence-transformers/natural-questions")["train"]
    data = data.train_test_split(test_size=0.2, seed=42, shuffle=True)

    train_answers = data["train"]["answer"]

    test_queries = data["test"]["query"]
    test_answers = data["test"]["answer"]
    target = torch.arange(len(test_answers))

    tf_idf.fit(train_answers)

    predict = tf_idf.retrieve(test_answers, test_queries)

    print(f"Recall@1: {recall_at_k(target, predict, k=1):.4f}",
          f"Recall@3: {recall_at_k(target, predict, k=3):.4f}",
          f"Recall@10: {recall_at_k(target, predict, k=10):.4f}",
          f"MRR: {mrr(target, predict):.4f}", sep="\n")
