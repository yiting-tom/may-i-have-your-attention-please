from typing import Dict, Any, List
import torch
from pathlib import Path
from torch.utils.data import Dataset
from transformers import BertTokenizer
from datasets import load_dataset


class SST2Dataset(Dataset):
    def __init__(self, tokenizer=BertTokenizer):
        self.data = self.load_data()
        self.tokenizer = tokenizer
        self.info = self.tokenize_process(
            tokenizer=tokenizer,
            sentences=self.data["sentence"],
        )
        self.max_sent_len = self.info["max_len"]
        self.input_ids = self.info["input_ids"]
        self.attn_mask = self.info["attention_mask"]

    def load_data(self):
        return load_dataset("glue", "sst2")["train"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return {
            "label": self.data["label"][idx],
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attn_mask[idx],
        }

    def tokenize_process(
        self, tokenizer: BertTokenizer, sentences: List[str]
    ) -> Dict[str, Any]:
        max_len = max([len(tokenizer.encode(sent)) for sent in sentences])
        input_ids = []
        attn_mask = []

        for sent in sentences:
            encode_dict = tokenizer.encode_plus(
                text=sent,
                add_special_tokens=True,
                max_length=max_len,
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors="pt",  # Return pytorch tensors.
            )
            input_ids.append(encode_dict["input_ids"])
            attn_mask.append(encode_dict["attention_mask"])

        return {
            "max_len": max_len,
            "input_ids": torch.cat(input_ids, dim=0),
            "attention_mask": torch.cat(attn_mask, dim=0),
        }


class ColaDataset(Dataset):
    def __init__(self, data_path: Path, tokenizer=BertTokenizer):
        self.data_path = data_path
        self.data = self.load_data()
        self.tokenizer = tokenizer
        self.info = self.tokenize_process(
            tokenizer=tokenizer,
            sentences=self.data,
        )
        self.max_sent_len = self.info["max_len"]
        self.input_ids = self.info["input_ids"]
        self.attn_mask = self.info["attention_mask"]

    def load_data(self):
        return open(self.data_path, "r").read().splitlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        line = self.data[idx]
        line = line.split("\t")
        label = int(line[1])
        text = line[3]
        return {
            "label": label,
            "text": text,
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attn_mask[idx],
        }

    def tokenize_process(
        self, tokenizer: BertTokenizer, sentences: List[str]
    ) -> Dict[str, Any]:
        max_len = max([len(tokenizer.encode(sent)) for sent in sentences])
        input_ids = []
        attn_mask = []

        for sent in sentences:
            encode_dict = tokenizer.encode_plus(
                text=sent,
                add_special_tokens=True,
                max_length=max_len,
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors="pt",  # Return pytorch tensors.
            )
            input_ids.append(encode_dict["input_ids"])
            attn_mask.append(encode_dict["attention_mask"])

        return {
            "max_len": max_len,
            "input_ids": torch.cat(input_ids, dim=0),
            "attention_mask": torch.cat(attn_mask, dim=0),
        }
