import torch.nn as nn
from transformers import AutoModel


class TransformerModel(nn.Module):
    def __init__(self, model_name, model_dir, dropout, n_fc, n_classes):
        super(TransformerModel, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name, cache_dir=model_dir)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(n_fc, n_classes)

    def forward(self, ids, mask, token_type_ids):

        hidden_output, pooled_output = self.transformer(
            ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False
        )
        pooled_output = self.drop(pooled_output)
        output = self.classifier(pooled_output)
        return output
