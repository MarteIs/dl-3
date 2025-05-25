from retrieval.trainer import ModelTrainer
from retrieval.hard_negatives import HardNegativeMiner

class HardNegativeTrainer(ModelTrainer):
    def __init__(self, cfg, model, loss_fn):
        super().__init__(cfg, model, loss_fn)
        self.miner = HardNegativeMiner(model)
    
    def train_step(self, batch):
        anchors = batch['anchor']
        positives = batch['positive']
        
        neg_indices = self.miner.find_negatives(anchors, positives)
        negatives = positives[neg_indices]
        
        loss = self.loss_fn(anchors, positives, negatives)
        loss.backward()
        return loss.item()