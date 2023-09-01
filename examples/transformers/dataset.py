import torch


class TransformerDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        text = " ".join(text.split())
        inputs = self.tokenizer(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        inputs = {k: v.type(torch.long).squeeze(0) for k, v in inputs.items()}

        labels_pt = torch.tensor(self.labels[idx], dtype=torch.float)
        return inputs, labels_pt

    def __len__(self):
        return len(self.labels)
