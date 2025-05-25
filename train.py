import click
import os
from pathlib import Path
from retrieval.trainable import RetrievalTrainable
from retrieval.config import RetrievalConfig
from retrieval.transformer_retriever import TransformerRetriever, RetrievalModel
from retrieval.trainer import Trainer
from retrieval.data import RetrievalCollator, load_data
import torch
import wandb
import yaml

@click.command()
@click.option('--config-path', type=Path, required=True)
@click.option("--wandb_key", type=str, required=False)
def main(config_path: Path,wandb_key: str = ""):
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    config = RetrievalConfig(**data)

    transformer = RetrievalModel(config)

    trainable = RetrievalTrainable(config.trainer)
    trainer = Trainer(config.trainer, transformer, trainable, RetrievalCollator())

    sims = None
    if config.hard_negatives:
        transformer = TransformerRetriever(config)

        train_dataset, val_dataset = load_data(config, test=False, scale=0.7)

        train_sims = transformer.retrieve(train_dataset, return_indices=False)
        val_sims = transformer.retrieve(val_dataset, return_indices=False)

        sims = (train_sims, val_sims)

        # torch.save(train_sims, "train_sims.pt")
        # torch.save(val_sims, "val_sims.pt")

    if config.trainer.use_wandb:
        if not wandb_key:
            raise RuntimeError("Wandb usage is turned on but wandb key wasn't passed.")

        os.environ['WANDB_API_KEY'] = wandb_key
        wandb.login()

        wandb.init(
            project=f"{config.trainer.experiment_name}",
            name=f"{config.trainer.experiment_name}" + "_run")

    train_dataset, val_dataset = load_data(config, test=False, sims=sims, scale=0.7)
    trainer.train(train_dataset, val_dataset)

if __name__ == "__main__":
    main()
