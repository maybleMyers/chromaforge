import torch
from typing import List

class QwenTextProcessingEngine:
    def __init__(self, text_encoder, tokenizer, emphasis_name, min_length=1):
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.emphasis_name = emphasis_name
        self.min_length = min_length

    def __call__(self, prompts: List[str], return_attention_mask=False):
        # Tokenize
        inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )

        # Move to text encoder device
        device = self.text_encoder.device
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        # Encode
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        embeddings = outputs.last_hidden_state

        if return_attention_mask:
            return embeddings, attention_mask
        return embeddings

    def tokenize(self, prompts: List[str]):
        return self.tokenizer(prompts, return_tensors="pt").input_ids
