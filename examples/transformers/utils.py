import torch
from dataset import TransformerDataset
from datasets import load_dataset
from model import TransformerModel
from transformers import AutoTokenizer

from ignite.handlers import DiskSaver


def get_tokenizer(tokenizer_name, tokenizer_dir):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=tokenizer_dir, do_lower_case=True)
    return tokenizer


def get_model(model_name, model_dir, drop_out, n_fc, num_classes):
    model = TransformerModel(model_name, model_dir, drop_out, n_fc, num_classes)
    return model


def get_dataset(cache_dir, tokenizer_name, tokenizer_dir, max_length):
    train_dataset, test_dataset = load_dataset("imdb", split=["train", "test"], cache_dir=cache_dir)
    tokenizer = get_tokenizer(tokenizer_name, tokenizer_dir)
    train_texts, train_labels = train_dataset["text"], train_dataset["label"]
    test_texts, test_labels = test_dataset["text"], test_dataset["label"]
    train_dataset = TransformerDataset(train_texts, train_labels, tokenizer, max_length)
    test_dataset = TransformerDataset(test_texts, test_labels, tokenizer, max_length)
    return train_dataset, test_dataset


def thresholded_output_transform(output):
    y_pred, y = output
    return torch.round(torch.sigmoid(y_pred)), y


def get_save_handler(config):
    if config["with_clearml"]:
        from ignite.handlers.clearml_logger import ClearMLSaver

        return ClearMLSaver(dirname=config["output_dir"])

    return DiskSaver(config["output_dir"], require_empty=False)
