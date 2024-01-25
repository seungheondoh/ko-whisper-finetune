import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import evaluate
import whisper

class FinetuneModel(nn.Module):
    def __init__(self, model_name, tokenizer):
        super(FinetuneModel, self).__init__()
        self.model = whisper.load_model(model_name)
        # only decoder training
        for p in self.model.encoder.parameters():
            p.requires_grad = False
        self.tokenizer = tokenizer
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.metrics_wer = evaluate.load("wer")
        self.metrics_cer = evaluate.load("cer")
        
    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(self, mel, tokens):
        return self.model.decoder(tokens, self.model.encoder(mel))

    def training_step(self, batch):
        mels = batch["mels"]
        labels = batch["labels"].long()
        tokens = batch["tokens"].long()
        with torch.no_grad(): # freeze encoder
            audio_features = self.model.encoder(mels)
            audio_features = audio_features.detach()
        out = self.model.decoder(tokens, audio_features)
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1)) # langauge model loss
        return loss
    
    def validation_step(self, batch):
        mels = batch["mels"]
        labels = batch["labels"].long()
        tokens = batch["tokens"].long()
        with torch.no_grad(): # inference only
            audio_features = self.model.encoder(mels)
            out = self.model.decoder(tokens, audio_features)
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))

        out[out == -100] = self.tokenizer.eot
        labels[labels == -100] = self.tokenizer.eot

        o_list, l_list = [], []
        for o, l in zip(out, labels):
            o = torch.argmax(o, dim=1) # greedy decoding
            o_list.append(self.tokenizer.decode(o, skip_special_tokens=True))
            l_list.append(self.tokenizer.decode(l, skip_special_tokens=True))
        cer = self.metrics_cer.compute(references=l_list, predictions=o_list)
        wer = self.metrics_wer.compute(references=l_list, predictions=o_list)
        return {
            "cer": cer,
            "wer": wer,
            "loss": loss
        }
