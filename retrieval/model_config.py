from pydantic import BaseModel, Field
from typing import Tuple, Annotated, Literal, Optional

class BaseOptimizerConfig(BaseModel):
    learning_rate: float


class AdamConfig(BaseOptimizerConfig):
    optimizer: Literal['adam'] = 'adam'


class AdamWConfig(BaseOptimizerConfig):
    optimizer: Literal['adamw'] = 'adamw'

    weight_decay: float


class SGDConfig(BaseOptimizerConfig):
    optimizer: Literal['sgd'] = 'sgd'


AnyOptimizerConfig = Annotated[AdamConfig | AdamWConfig | SGDConfig, Field(discriminator='optimizer')]


class LinearWarmupSchedulerConfig(BaseModel):
    schedule: Literal['linear_warmup'] = 'linear_warmup'
    warmup_steps_proportion: float


class ConstantSchedulerConfig(BaseModel):
    schedule: Literal['constant'] = 'constant'


AnySchedulerConfig = Annotated[LinearWarmupSchedulerConfig | ConstantSchedulerConfig, Field(discriminator='schedule')]

class BaseLossConfig(BaseModel):
    similarity_margin: float

class TripletLossConfig(BaseLossConfig):
    loss: Literal["triplet"] = "triplet"


class ContrastiveLossConfig(BaseLossConfig):
    loss: Literal["contrastive"] = "contrastive"



AnyLossConfig = Annotated[TripletLossConfig | ContrastiveLossConfig, Field(discriminator='loss')]

class TrainerConfig(BaseModel):
    num_epochs: int
    gradient_accumulation_steps: int
    project_dir: str
    log_steps: int
    eval_steps: int
    save_steps: int
    experiment_name: str
    log_with: str
    batch_size: int
    shuffle_train_dataset: bool
    num_workers: int
    seed: int
    use_wandb: bool

    optimizer: AnyOptimizerConfig
    scheduler: AnySchedulerConfig
    loss: AnyLossConfig


class RetrievalConfig(BaseModel):
    trainer: TrainerConfig
    max_length: int
    base_model: str
    dataset: str
    query_prefix: str
    document_prefix: str
    use_contrastive_format: bool
    hard_negatives: Optional[bool] = False