import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import AutoTokenizer
from pyvi import ViTokenizer
import os

from utils import load_json, save_json


class CustomDataset(Dataset):
    def __init__(self,
        data_path: str,
        label_type: str,
        tokenizer_name: str = None,
        cache_dir: str = None,
        vocab_folder_path: str = None,
        max_len: int = 128,
    ):
        super(CustomDataset, self).__init__()

        self.data = load_json(data_path)
        self.max_len = max_len
        
        if tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
        else:
            if vocab_folder_path:
                if not (
                    os.path.exists(os.path.join(vocab_folder_path, "vocab.json"))
                    and os.path.exists(os.path.join(vocab_folder_path, "word2idx.json"))
                ):
                    self.create_vocab_and_word2idx(vocab_folder_path)
                else:
                    self.word2idx = load_json(os.path.join(vocab_folder_path, "word2idx.json"))
            else:
                raise ValueError("word2idx_path is required.")
            self.tokenizer = None

        self.label_type = label_type
        if self.label_type == "toxicity":
            self.label_dict = {"NON-TOXIC": 0, "TOXIC": 1}
        elif self.label_type == "toxic_type":
            self.label_dict = {"OFFENSIVE": 0, "HATE": 1}
        elif self.label_type == "expression_type":
            self.label_dict = {"IMPLICIT": 0, "EXPLICIT": 1, "REPORT": 2}
        else:
            raise ValueError(f"Unsupported label type: {self.label_type}. Supported types are: toxicity, toxic_type, expression_type.")
        
    def __len__(self):
        """Get length of dataset."""
        return len(self.data)

    def __getitem__(self, index: int):
        """Get item by index."""
        item = self.data[index]
        
        label_name = item[self.label_type]
        label = self.label_dict[label_name]

        text = item["text"]
        if self.tokenizer:
            if self.tokenizer.name_or_path in [
                "vinai/phobert-base-v2",
                "Fsoft-AIC/videberta-base",
                "vinai/bartpho-word"
            ]:
                text = ViTokenizer.tokenize(text)

            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt"
            )
            
            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)

        else:
            text = ViTokenizer.tokenize(text)
            
            input_ids = []
            for word in text.split(" "):
                input_ids.append(self.word2idx.get(word, self.word2idx["<unk>"]))
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            input_ids = F.pad(
                input_ids,
                (0, self.max_len - input_ids.size(0)),
                value=0
            )

            attention_mask = (input_ids != 0).long()
            
        return {
            "text": text,
            "label_name": label_name,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.long)
        }
    
    def create_vocab_and_word2idx(self, vocab_folder_path: str):
        """Create vocab and word to index."""
        vocab = set()
        for item in self.data:
            segmented_text = ViTokenizer.tokenize(item["text"])
            for segmented_word in segmented_text.split(" "):
                if segmented_word not in vocab:
                    vocab.add(segmented_word)
        vocab = list(vocab)
        save_json(vocab, os.path.join(vocab_folder_path, "vocab.json"))

        self.word2idx = {}
        self.word2idx["<pad>"] = 0
        for idx, word in enumerate(vocab, start=1):
            self.word2idx[word] = idx
        self.word2idx["<unk>"] = len(vocab) + 1
        save_json(self.word2idx, os.path.join(vocab_folder_path, "word2idx.json"))

        idx2word = {idx: word for word, idx in self.word2idx.items()}
        save_json(idx2word, os.path.join(vocab_folder_path, "idx2word.json"))
