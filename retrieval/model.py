from accelerate import Accelerator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModel
from tqdm import tqdm
from retrieval.data import RetrievalCollator
from retrieval.config import RetrievalConfig

class RetrievalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self._base_model = AutoModel.from_pretrained(self._config.base_model)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self._base_model(input_ids=input_ids, attention_mask=attention_mask)
        return F.normalize(outputs.pooler_output, dim=-1)

class TransformerRetriever:
    def __init__(self, config: RetrievalConfig, model: nn.Module=None):
        if not model:
            self._model = RetrievalModel(config)
        else:
            self._model = model
        self._config = config
        self._accelarator = Accelerator()
        self._data = None

    def _vectorize(self):
        if not self._data:
            raise ValueError()

        with torch.inference_mode():
            model, data = self._accelarator.prepare(self._model, self._data)

            document_vectors_total = []
            query_vectors_total = []

            for batch in tqdm(data, desc="Vectorizing"):
                query_vectors = model(batch["positive"]["input_ids"], batch["positive"]["attention_mask"])
                document_vectors = model(batch["anchor"]["input_ids"], batch["anchor"]["attention_mask"])

                query_vectors_total.append(query_vectors)
                document_vectors_total.append(document_vectors)

            query_vectors = torch.cat(query_vectors_total, dim=0)
            document_vectors = torch.cat(document_vectors_total, dim=0)

            return query_vectors, document_vectors

    def retrieve(self, dataset: Dataset, return_indices: bool=True):
        self._data = torch.utils.data.DataLoader(dataset, batch_size=self._config.trainer.batch_size, shuffle=False, pin_memory=True, persistent_workers=False,
                                      collate_fn=RetrievalCollator())

        query_vectors, document_vectors = self._vectorize()
        sims = query_vectors.cpu().detach() @ document_vectors.cpu().detach().T

        if return_indices:
            return sims.sort(dim=1, descending=True).indices

        return sims
