from retrieval.model import EmbeddingModel
from retrieval.data_loader import DataProcessor
from training.hard_negative_trainer import HardNegativeTrainer
from torch.nn import TripletMarginLoss

model = EmbeddingModel("intfloat/multilingual-e5-base")
processor = DataProcessor(config)
train_data = processor.load_data()

loss_fn = TripletMarginLoss(margin=0.2)
trainer = HardNegativeTrainer(config, model, loss_fn)

trainer.train(train_data)