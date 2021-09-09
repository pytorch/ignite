import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification


class TransformerModel(nn.Module):
    def __init__(self, model_name, model_dir, dropout, n_fc, n_classes):
        super(TransformerModel, self).__init__()
        self.config = AutoConfig.from_pretrained(
            model_name,
            num_labels=n_classes,
            output_hidden_states=n_fc,
            classifier_dropout=dropout,
            output_attentions=True,
        )
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            model_name, cache_dir=model_dir, config=self.config
        )

    def forward(self, inputs):
        output = self.transformer(**inputs)["logits"]

        return output
