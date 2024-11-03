"""Adapted from https://github.com/lm-sys/RouteLLM/"""

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class BERTRouter:
    def __init__(self, checkpoint_path: str, device) -> None:
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_path,
            num_labels=3,
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.model.eval()

    def calculate_strong_win_rate(self, prompt: str) -> float:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.cpu().numpy()[0]

        exp_scores = np.exp(logits - np.max(logits))
        softmax_scores = exp_scores / np.sum(exp_scores)

        # Compute prob of label 1 and 2 (tie, tier 2 wins)
        binary_prob = np.sum(softmax_scores[-2:])
        return 1 - binary_prob
