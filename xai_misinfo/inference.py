import os, json, torch, joblib
import numpy as np
from transformers import AutoTokenizer
from train_pipeline import TransformerClassifier, Vocab, RNNClassifier

class InferenceEngine:
    def __init__(self, models_dir='models', transformer_name='xlm-roberta-base'):
        self.dir = models_dir
        self.transformer_name = transformer_name
        self._load_artifacts()
    def _load_artifacts(self):
        # Load summary
        summary_path = os.path.join(self.dir,'model_summary.json')
        self.summary = json.load(open(summary_path)) if os.path.isfile(summary_path) else {}
        self.best = self.summary.get('best_model')
        # Load supporting assets
        self.vectorizer = None
        tfidf_path = os.path.join(self.dir,'tfidf.pkl')
        if os.path.isfile(tfidf_path):
            self.vectorizer = joblib.load(tfidf_path)
        # Load classical models
        self.classical = {}
        for name in ['svm','random_forest','xgboost']:
            path = os.path.join(self.dir,f'{name}.pkl')
            if os.path.isfile(path):
                self.classical[name] = joblib.load(path)
        # Load transformer
        # Load multiple transformers if present
        self.transformers = []  # list of (name, model, tokenizer)
        for file in os.listdir(self.dir):
            if file.startswith('transformer_') and file.endswith('.pt'):
                path = os.path.join(self.dir, file)
                ckpt = torch.load(path, map_location='cpu')
                model_name = ckpt.get('model_name', self.transformer_name)
                try:
                    tok = AutoTokenizer.from_pretrained(model_name)
                    mdl = TransformerClassifier(model_name)
                    mdl.load_state_dict(ckpt['model_state'])
                    mdl.eval()
                    self.transformers.append((model_name, mdl, tok))
                except Exception:
                    continue
    def predict(self, text: str):
        outputs = {}
        if self.vectorizer:
            vec = self.vectorizer.transform([text])
            for name, model in self.classical.items():
                try:
                    if hasattr(model,'predict_proba'):
                        prob = model.predict_proba(vec)[0][1]
                    else:
                        # fallback: decision function -> sigmoid-ish
                        d = model.decision_function(vec)
                        prob = float(1/(1+np.exp(-d)) if isinstance(d, np.ndarray) else 0.5)
                    outputs[name] = prob
                except Exception:
                    pass
        if self.transformers:
            import torch
            t_probs = []
            for model_name, mdl, tok in self.transformers:
                enc = tok(text, truncation=True, padding=True, return_tensors='pt', max_length=128)
                with torch.no_grad():
                    logits = mdl(enc['input_ids'], enc['attention_mask'])
                    prob = float(logits.softmax(-1)[0][1].cpu().numpy())
                outputs[f'transformer:{model_name}'] = prob
                t_probs.append(prob)
        # Choose best model probability
        if self.best and self.best in outputs:
            final_prob = outputs[self.best]
        else:
            final_prob = np.mean(list(outputs.values())) if outputs else 0.0
        label = 'Misleading' if final_prob >= 0.5 else ('I don\'t know' if 0.4 <= final_prob < 0.5 else 'Not Misleading')
        return {'probability': final_prob, 'label': label, 'model_outputs': outputs, 'best_model': self.best}
