import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

STOPWORDS_EN = set(stopwords.words('english'))

# ---------------- Classical Models ---------------- #

def build_classical_models(pos_weight_ratio=None):
    models = {
        'svm': LinearSVC(class_weight='balanced'),
        'random_forest': RandomForestClassifier(n_estimators=300, class_weight='balanced', n_jobs=-1, random_state=42),
        'xgboost': XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=pos_weight_ratio if pos_weight_ratio else 1.0
        )
    }
    return models

# ---------------- Deep Learning RNN Models ---------------- #
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer=None, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        t = self.texts[idx]
        if self.tokenizer:
            enc = self.tokenizer(t, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
            item = {k: v.squeeze(0) for k, v in enc.items()}
        else:
            item = {'text': t}
        item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, rnn_type='lstm', bidirectional=False, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        rnn_cls = {'lstm': nn.LSTM, 'gru': nn.GRU}[rnn_type]
        self.rnn = rnn_cls(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        fc_in = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(fc_in, num_classes)
    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        # use last timestep
        out = out[:, -1, :]
        return self.fc(out)

# ---------------- Transformer Fine-tuning ---------------- #
class TransformerClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.base = AutoModel.from_pretrained(model_name)
        hidden = self.base.config.hidden_size
        self.classifier = nn.Linear(hidden, 2)
    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:,0]
        return self.classifier(pooled)

# ---------------- Utility Functions ---------------- #

def clean_en(text: str):
    t = text.lower()
    tokens = [w for w in t.split() if w not in STOPWORDS_EN]
    return ' '.join(tokens)

def evaluate_preds(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    return {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1}

def save_metrics(name, metrics, out_dir):
    path = os.path.join(out_dir, f"metrics_{name}.json")
    with open(path,'w') as f:
        json.dump(metrics, f, indent=2)

# ---------------- Training Routines ---------------- #

def train_classical(X_train, y_train, X_val, y_val, out_dir, english_only=False):
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    # Estimate positive weight ratio for xgboost (neg/pos)
    import numpy as _np
    pos = sum(1 for v in y_train if v==1)
    neg = len(y_train)-pos
    ratio = (neg/pos) if pos>0 else 1.0
    for name, model in build_classical_models(pos_weight_ratio=ratio).items():
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_val_vec)
        metrics = evaluate_preds(y_val, preds)
        results[name] = metrics
        save_metrics(name, metrics, out_dir)
        joblib.dump(model, os.path.join(out_dir, f"{name}.pkl"))
    joblib.dump(vectorizer, os.path.join(out_dir, 'tfidf.pkl'))
    return results

# Placeholder simple token indexer for RNNs (for demonstrative baseline)
class Vocab:
    def __init__(self):
        self.idx = {'<pad>':0, '<unk>':1}
    def build(self, texts, max_size=50000):
        from collections import Counter
        c = Counter()
        for t in texts:
            for w in t.split():
                c[w]+=1
        for w,_ in c.most_common(max_size-len(self.idx)):
            self.idx.setdefault(w, len(self.idx))
    def encode(self, text, max_len=100):
        ids = [self.idx.get(w,1) for w in text.split()][:max_len]
        if len(ids)<max_len:
            ids += [0]*(max_len-len(ids))
        return ids
    @property
    def size(self):
        return len(self.idx)

def train_rnn(texts_train, y_train, texts_val, y_val, out_dir, rnn_type='lstm', bidirectional=False, epochs=3, batch_size=64):
    vocab = Vocab(); vocab.build(texts_train)
    def enc_batch(texts):
        return torch.tensor([vocab.encode(t) for t in texts], dtype=torch.long)
    model = RNNClassifier(vocab_size=vocab.size, rnn_type=rnn_type, bidirectional=bidirectional)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(epochs):
        model.train()
        for i in range(0, len(texts_train), batch_size):
            batch_texts = texts_train[i:i+batch_size]
            batch_labels = y_train[i:i+batch_size]
            inputs = enc_batch(batch_texts)
            labels = torch.tensor(batch_labels)
            logits = model(inputs)
            loss = criterion(logits, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        # simple val
    model.eval()
    with torch.no_grad():
        val_inputs = enc_batch(texts_val)
        val_logits = model(val_inputs)
        preds = val_logits.argmax(-1).numpy()
    metrics = evaluate_preds(y_val, preds)
    os.makedirs(out_dir, exist_ok=True)
    torch.save({'model_state': model.state_dict(), 'vocab': vocab.idx}, os.path.join(out_dir, f"rnn_{rnn_type}{'_bi' if model.rnn.bidirectional else ''}.pt"))
    save_metrics(f"rnn_{rnn_type}", metrics, out_dir)
    return metrics


def train_transformer(texts_train, y_train, texts_val, y_val, out_dir, model_name='xlm-roberta-base', epochs=1, batch_size=8):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TransformerClassifier(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train_ds = TextDataset(texts_train, y_train, tokenizer)
    val_ds = TextDataset(texts_val, y_val, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    optim = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Transformer epoch {ep+1}"):
            input_ids = batch['input_ids'].to(device); attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            optim.zero_grad(); loss.backward(); optim.step()
    # validation
    model.eval(); preds=[]
    with torch.no_grad():
        for batch in val_loader:
            logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            preds.extend(logits.argmax(-1).cpu().numpy())
    metrics = evaluate_preds(y_val, preds)
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained = None  # disable accidental huggingface style call
    # Sanitize model name for filename
    slug = model_name.replace('/', '_')
    torch.save({'model_state': model.state_dict(), 'model_name': model_name}, os.path.join(out_dir, f'transformer_{slug}.pt'))
    # Save tokenizer for inference
    tok_dir = os.path.join(out_dir, f'tokenizer_{slug}')
    try:
        tokenizer.save_pretrained(tok_dir)
    except Exception:
        pass
    save_metrics(f'transformer_{slug}', metrics, out_dir)
    return metrics, tokenizer, slug

# ---------------- CNN Text Classifier ---------------- #
class CNNTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_classes=2, kernel_sizes=(3,4,5), num_filters=64, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)
    def forward(self, x):
        emb = self.embedding(x)  # (B,L,E)
        emb = emb.transpose(1,2)  # (B,E,L)
        conv_outs = [torch.relu(conv(emb)).max(dim=2)[0] for conv in self.convs]
        cat = torch.cat(conv_outs, dim=1)
        cat = self.dropout(cat)
        return self.fc(cat)

def train_cnn(texts_train, y_train, texts_val, y_val, out_dir, epochs=3, batch_size=64, max_len=100):
    vocab = Vocab(); vocab.build(texts_train)
    def enc_batch(texts):
        return torch.tensor([vocab.encode(t, max_len=max_len) for t in texts], dtype=torch.long)
    model = CNNTextClassifier(vocab_size=vocab.size)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(epochs):
        model.train()
        for i in range(0, len(texts_train), batch_size):
            bt = texts_train[i:i+batch_size]
            bl = y_train[i:i+batch_size]
            inputs = enc_batch(bt)
            labels = torch.tensor(bl)
            logits = model(inputs)
            loss = criterion(logits, labels)
            optim.zero_grad(); loss.backward(); optim.step()
    model.eval()
    with torch.no_grad():
        val_inputs = enc_batch(texts_val)
        val_logits = model(val_inputs)
        preds = val_logits.argmax(-1).numpy()
    metrics = evaluate_preds(y_val, preds)
    os.makedirs(out_dir, exist_ok=True)
    torch.save({'model_state': model.state_dict(), 'vocab': vocab.idx}, os.path.join(out_dir, 'cnn.pt'))
    save_metrics('cnn', metrics, out_dir)
    return metrics

# ---------------- Main Pipeline ---------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='Path to CSV with text,label columns')
    ap.add_argument('--misinfo_csv', help='Optional separate CSV containing misinfo texts (single column text)')
    ap.add_argument('--nonmisinfo_csv', help='Optional separate CSV containing nonmisinfo texts (single column text)')
    ap.add_argument('--transformer_model', default='xlm-roberta-base', help='(Deprecated if --transformer_models used) single transformer model name')
    ap.add_argument('--transformer_models', default='', help='Comma-separated list of transformer models to fine-tune (e.g. xlm-roberta-base,distilbert-base-multilingual-cased,bert-base-multilingual-cased)')
    ap.add_argument('--kfold', type=int, default=0, help='If >0 use stratified K-fold (k) instead of single split')
    ap.add_argument('--val_size', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--transformer_epochs', type=int, default=1)
    ap.add_argument('--rnn_epochs', type=int, default=3)
    ap.add_argument('--cnn_epochs', type=int, default=3)
    ap.add_argument('--max_per_class', type=int, default=0, help='If >0, randomly sample at most this many examples per class for faster prototyping')
    args = ap.parse_args()

    np.random.seed(args.seed)

    # Build dataframe from either combined file or two separate files
    if args.misinfo_csv and args.nonmisinfo_csv:
        mis_df = pd.read_csv(args.misinfo_csv)
        non_df = pd.read_csv(args.nonmisinfo_csv)
        # assume first column is text if no header 'text'
        if 'text' not in mis_df.columns:
            mis_df.columns = ['text']
        if 'text' not in non_df.columns:
            non_df.columns = ['text']
        mis_df['label'] = 1
        non_df['label'] = 0
        df = pd.concat([mis_df[['text','label']], non_df[['text','label']]], ignore_index=True)
    else:
        df = pd.read_csv(args.data)
        assert 'text' in df.columns and 'label' in df.columns, 'CSV must contain text,label OR provide --misinfo_csv & --nonmisinfo_csv'

    # Normalize label strings if present
    if df['label'].dtype == object:
        mapping = {'misinformation':1,'misinfo':1,'misleading':1,'nonmisinfo':0,'nonmisinformation':0,'not_misleading':0,'legit':0,'true':0,'false':1}
        df['label'] = df['label'].str.lower().map(mapping)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    # Basic English cleaning variant (not applied to whole multilingual set to keep tokens)
    texts = df['text'].astype(str).tolist()
    labels = df['label'].astype(int).tolist()

    # Optional down-sampling for faster experimentation
    if args.max_per_class and args.max_per_class > 0:
        # Stratified sample per class
        sampled_frames = []
        for cls in [0,1]:
            sub = df[df['label']==cls]
            if len(sub) > args.max_per_class:
                sub = sub.sample(args.max_per_class, random_state=args.seed)
            sampled_frames.append(sub)
        df = pd.concat(sampled_frames).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    X_train, X_val, y_train, y_val = train_test_split(df['text'].tolist(), df['label'].tolist(), test_size=args.val_size, stratify=df['label'].tolist(), random_state=args.seed)

    out_dir = 'models'
    os.makedirs(out_dir, exist_ok=True)

    print('\n== Training Classical Models ==')
    classical_metrics = train_classical(X_train, y_train, X_val, y_val, out_dir)

    print('\n== Training RNN Baselines ==')
    rnn_lstm_metrics = train_rnn(X_train, y_train, X_val, y_val, out_dir, rnn_type='lstm', epochs=args.rnn_epochs)
    rnn_gru_metrics = train_rnn(X_train, y_train, X_val, y_val, out_dir, rnn_type='gru', epochs=args.rnn_epochs)
    rnn_bigru_metrics = train_rnn(X_train, y_train, X_val, y_val, out_dir, rnn_type='gru', bidirectional=True, epochs=args.rnn_epochs)

    print('\n== Training CNN Text Classifier ==')
    cnn_metrics = train_cnn(X_train, y_train, X_val, y_val, out_dir, epochs=args.cnn_epochs)

    transformer_collected = []
    model_list = []
    if args.transformer_models.strip():
        model_list = [m.strip() for m in args.transformer_models.split(',') if m.strip()]
    else:
        model_list = [args.transformer_model]
    print('\n== Fine-tuning Transformer Models ==')
    for tm in model_list:
        print(f'-- Training {tm}')
        t_metrics, tokenizer, slug = train_transformer(X_train, y_train, X_val, y_val, out_dir, model_name=tm, epochs=args.transformer_epochs)
        transformer_collected.append((f'transformer_{slug}', t_metrics))

    # Select best by F1
    collected = []
    def f1(m): return m.get('f1',0)
    for name, m in classical_metrics.items():
        collected.append((name, f1(m), m))
    collected.append(('rnn_lstm', f1(rnn_lstm_metrics), rnn_lstm_metrics))
    collected.append(('rnn_gru', f1(rnn_gru_metrics), rnn_gru_metrics))
    collected.append(('rnn_bigru', f1(rnn_bigru_metrics), rnn_bigru_metrics))
    for name, met in transformer_collected:
        collected.append((name, f1(met), met))
    collected.append(('cnn', f1(cnn_metrics), cnn_metrics))
    best = max(collected, key=lambda x: x[1])

    # Identify top 2 neural models among rnn_* and cnn
    neural_candidates = [(n, f, m) for (n,f,m) in collected if n.startswith('rnn_') or n=='cnn']
    neural_sorted = sorted(neural_candidates, key=lambda x: x[1], reverse=True)
    top_neural = [n for n,_,_ in neural_sorted[:2]]

    summary = {
        'best_model': best[0],
        'best_f1': best[1],
        'top_neural_models': top_neural,
        'all': {name:met for name,_,met in collected}
    }
    with open(os.path.join(out_dir,'model_summary.json'),'w') as f:
        json.dump(summary,f,indent=2)
    print('\nBest model:', best[0], 'F1=', best[1])

if __name__ == '__main__':
    main()
