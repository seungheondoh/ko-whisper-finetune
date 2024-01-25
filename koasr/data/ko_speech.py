import os
import torch
import whisper
import numpy as np
from torch.utils.data import Dataset
from datasets import load_dataset

class MelDataset(Dataset):
    def __init__(self, split, tokenizer):
        self.split = split
        self.tokenizer = tokenizer
        self._load_dataset()
        
    def _load_dataset(self):
        dataset = load_dataset("NX2411/AIhub-korean-speech-data")
        if self.split == "train":
            self.dataset = dataset["train"]
        else:
            self.dataset = dataset["test"]

    def __getitem__(self, index):
        item = self.dataset[index]
        audio = item['input_values']
        text = item['texts']
        
        audio = whisper.pad_or_trim(np.array(audio).astype(np.float32).flatten())
        mel = whisper.log_mel_spectrogram(audio)
        token = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(text)
        label = token[1:] + [self.tokenizer.eot]
        return {
            "mel": mel,
            "token": token,
            "label": label
        }

    def __len__(self):
        return len(self.dataset)

class WhisperDataCollatorWhithPadding:
    def __call__(sefl, features):
        mels, tokens, labels = [], [], []
        for f in features:
            mels.append(f["mel"])
            tokens.append(f["token"])
            labels.append(f["label"])

        mels = torch.concat([input_id[None, :] for input_id in mels])
        label_lengths = [len(lab) for lab in labels]
        tokens_length = [len(e) for e in tokens]
        max_label_len = max(label_lengths+tokens_length)
        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in zip(labels, label_lengths)]
        tokens = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) for e, e_len in zip(tokens, tokens_length)] # 50257 is eot token id
        batch = {
            "labels": labels,
            "tokens": tokens
        }
        batch = {k: torch.tensor(np.array(v)) for k, v in batch.items()}
        batch["mels"] = mels
        return batch

