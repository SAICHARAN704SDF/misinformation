import shap, torch, numpy as np
from typing import List

# Generic wrapper for transformer or classical model shap explanation

def explain_transformer(model, tokenizer, texts: List[str], max_samples=10):
    sample_texts = texts[:max_samples]
    def f(batch):
        enc = tokenizer(batch, truncation=True, padding=True, return_tensors='pt', max_length=128)
        with torch.no_grad():
            logits = model(enc['input_ids'], enc['attention_mask'])
        return logits.softmax(-1).numpy()
    explainer = shap.Explainer(f, shap.maskers.Text(tokenizer))
    return explainer(sample_texts)
