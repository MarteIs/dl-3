from abc import ABC, abstractmethod
from pathlib import Path
import torchmetrics
from tqdm import tqdm
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from typing import Any
from torch.utils.data import Dataset, DataLoader
from torch.optim.optimizer import ParamsT
from torch.optim import Optimizer, SGD, Adam, AdamW
from torch.optim.lr_scheduler import LRScheduler
from retrieval.config import TrainerConfig, SGDConfig, AdamConfig, AdamWConfig, ConstantSchedulerConfig, \
    LinearWarmupSchedulerConfig
import wandb

class Trainable(ABC):
    @abstractmethod
    def forward_pass(self, model: nn.Module, model_inputs) -> tuple[torch.Tensor, Any]:
        ...

    @abstractmethod
    def create_metrics(self) -> dict[str, torchmetrics.Metric]:
        ...

    @abstractmethod
    def update_metrics(self, model_outputs, metrics: dict[str, torchmetrics.Metric]):
        ...


class Trainer:
    def __init__(self, config: TrainerConfig, model: nn.Module, trainable: Trainable, collator):
        self._config = config
        self._model = model
        self._trainable = trainable
        self._collator = collator

    def _setup_optimizer(self, params: ParamsT) -> Optimizer:
        match self._config.optimizer:
            case SGDConfig():
                return SGD(params, lr=self._config.optimizer.learning_rate)
            case AdamConfig():
                return Adam(params, lr=self._config.optimizer.learning_rate)
            case AdamWConfig():
                return AdamW(
                    params,
                    lr=self._config.optimizer.learning_rate,
                    weight_decay=self._config.optimizer.weight_decay
                )
            case _:
                raise ValueError(f'Optimizer {self._config.optimizer} is not supported')

    def _setup_scheduler(self, optimizer: Optimizer, num_total_steps: int) -> LRScheduler:
        match self._config.scheduler:
            case ConstantSchedulerConfig():
                return transformers.get_constant_schedule(optimizer)
            case LinearWarmupSchedulerConfig():
                return transformers.get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=int(num_total_steps * self._config.scheduler.warmup_steps_proportion),
                    num_training_steps=num_total_steps
                )
            case _:
                raise ValueError(f'Scheduler {self._config.scheduler} is not supported')

    def _setup_dataloader(self, dataset: Dataset, training: bool):
        shuffle = self._config.shuffle_train_dataset if training else False

        return DataLoader(
            dataset,
            batch_size=self._config.batch_size,
            shuffle=shuffle,
            num_workers=self._config.num_workers,
            collate_fn=self._collator,
            pin_memory=True,
            persistent_workers=True
        )

    def _compute_and_log_metrics(
            self,
            prefix: str,
            accelerator: Accelerator,
            metrics: dict[str, torchmetrics.Metric],
            current_step: int
    ):
        metric_values = {k: v.compute().item() for k, v in metrics.items()}
        metric_values = {f'{prefix}{k}': v for k, v in metric_values.items()}
        accelerator.log(metric_values, step=current_step)

    def _train_loop_iter(
            self,
            epoch_i: int,
            dataloader: DataLoader,
            dataloader_eval: DataLoader,
            model: nn.Module,
            optimizer: Optimizer,
            scheduler: LRScheduler,
            accelerator: Accelerator,
            pbar: tqdm
    ):
        model = model
        model.train()

        for i, minibatch in enumerate(dataloader):
            current_global_step = epoch_i * len(dataloader) + i
            current_global_optimizer_step = current_global_step // self._config.gradient_accumulation_steps

            train_loss = 0.
            with accelerator.accumulate():
                loss_value, outputs = self._trainable.forward_pass(model, minibatch)
                train_loss += loss_value

                accelerator.backward(loss_value)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss /= len(dataloader)

            if self._config.use_wandb:
                wandb_metrics = {"train_loss": train_loss}

                wandb.log(wandb_metrics)

            if accelerator.sync_gradients and \
                    current_global_optimizer_step != 0 and \
                    current_global_optimizer_step % self._config.eval_steps == 0:
                self._eval_loop_iter(dataloader_eval, model, accelerator, current_global_optimizer_step)
            # if accelerator.sync_gradients
            if current_global_optimizer_step != 0 and current_global_optimizer_step % self._config.save_steps == 0:
                accelerator.save_model(model, Path(self._config.project_dir) / f'save-{current_global_optimizer_step}')
                wandb.save(Path(self._config.project_dir) / f'save-{current_global_optimizer_step}')

            pbar.update(1)

    def _create_metrics(self, accelerator: Accelerator):
        metrics = self._trainable.create_metrics()
        metrics = {k: v.to(accelerator.device) for k, v in metrics.items()}
        return metrics


    def _eval_loop_iter(
            self,
            dataloader: DataLoader,
            model: nn.Module,
            accelerator: Accelerator,
            current_iter: int
    ):
        with torch.no_grad():
            model.eval()

            val_loss = 0.
            for minibatch in dataloader:
                #TODO: delete outputs
                loss_value, outputs = self._trainable.forward_pass(model, minibatch)

                val_loss += loss_value

            val_loss /= len(dataloader)

            if self._config.use_wandb:
                wandb_metrics = {"val_loss": val_loss}

                wandb.log(wandb_metrics)

            model.train()

    def train(self, train_dataset: Dataset, val_dataset: Dataset):
        set_seed(self._config.seed)
        train_dataloader = self._setup_dataloader(train_dataset, training=True)
        val_dataloader = self._setup_dataloader(val_dataset, training=False)
        accel = Accelerator(
            gradient_accumulation_steps=self._config.gradient_accumulation_steps,
            project_dir=self._config.project_dir,
            log_with=self._config.log_with
        )
        accel.init_trackers(
            self._config.experiment_name,
            config={
                'lr': self._config.optimizer.learning_rate,
            }
        )
        train_dataloader, val_dataloader = accel.prepare(train_dataloader, val_dataloader)
        num_total_steps = len(train_dataloader) * self._config.num_epochs
        optimizer = self._setup_optimizer(self._model.parameters())
        scheduler = self._setup_scheduler(optimizer, num_total_steps=num_total_steps)
        model, optimizer, scheduler = accel.prepare(self._model, optimizer, scheduler)
        with tqdm(total=self._config.num_epochs * len(train_dataloader)) as pbar:
            for epoch in range(self._config.num_epochs):
                self._train_loop_iter(
                    dataloader=train_dataloader,
                    dataloader_eval=val_dataloader,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    accelerator=accel,
                    epoch_i=epoch,
                    pbar=pbar
                )
        accel.end_training()
