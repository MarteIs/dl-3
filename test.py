import click
from pathlib import Path
from retrieval.trainable import RetrievalTrainable
from retrieval.config import RetrievalConfig
from retrieval.transformer_retriever import TransformerRetriever, RetrievalModel
from metrics import recall_at_k, mrr
from retrieval.trainer import Trainer
from retrieval.data import RetrievalCollator, load_data
import safetensors
import torch
import yaml

@click.command()
@click.option('--config-path', type=Path, required=True)
@click.option("--model-path", type=Path, required=False)
def main(config_path: Path, model_path: Path=None):
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    config = RetrievalConfig(**data)
    transformer = RetrievalModel(config)

    test_dataset = load_data(config, test=True)

    model = RetrievalModel(config)

    if model_path:
        safetensors.torch.load_model(model, model_path)

    retriever = TransformerRetriever(config, model)

    predict = retriever.retrieve(test_dataset)
    target = torch.arange(len(test_dataset))


    print(f"Recall@1: {recall_at_k(target, predict, k=1):.4f}",
          f"Recall@3: {recall_at_k(target, predict, k=3):.4f}",
          f"Recall@10: {recall_at_k(target, predict, k=10):.4f}",
          f"MRR: {mrr(target, predict):.4f}", sep="\n")

if __name__ == "__main__":
    main()