"""PHASE 1: Training script for a simple multilingual misinformation detector with XAI placeholders.
Meets teacher requirements:
1. Preprocess (English: lowercase, punctuation, stopwords removal).
2. 90/10 split.
3. Train RandomForest + XGBoost (choose best on F1).
4. Evaluate metrics (Accuracy, Precision, Recall, F1, Confusion Matrix).
5. Save best model + TF-IDF vectorizer.
6. Provide placeholder XAI explanation logic.
"""
import os
import re
import string
import argparse
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

# Try to import NLTK stopwords (fallback to basic list)
try:
    import nltk
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))
except Exception:
    STOPWORDS = {"the","a","an","is","to","and","in","of","for","on","with","at","by","this","that"}

PUNCT_TABLE = str.maketrans('', '', string.punctuation)

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    # lowercase
    text = text.lower()
    # remove urls
    text = re.sub(r"http[s]?://\S+", " ", text)
    # remove punctuation
    text = text.translate(PUNCT_TABLE)
    # remove digits
    text = re.sub(r"\d+", " ", text)
    # collapse whitespace
    tokens = [t for t in text.split() if t not in STOPWORDS]
    return " ".join(tokens)

def load_and_split_data(filepath: str) -> Tuple[List[str],List[str],List[int],List[int]]:
    if not os.path.isfile(filepath):
        # create dummy dataset
        data = {"text": [
            "Vaccines cause autism claims spread online",
            "NASA confirms new exoplanet discovery",
            "Miracle herbal cure for cancer revealed",
            "WHO releases new health guidelines",
            "Secret government plot exposed by anonymous source",
            "Scientists publish peer-reviewed climate change study"
        ], "label": [1,0,1,0,1,0]}
        pd.DataFrame(data).to_csv(filepath, index=False)
        print(f"Dummy dataset created at {filepath}")
    df = pd.read_csv(filepath)
    assert 'text' in df.columns and 'label' in df.columns, "CSV must have text,label columns"
    # preprocess english content only (non-English left as-is for simplicity)
    df['clean_text'] = df['text'].astype(str).apply(preprocess_text)
    X = df['clean_text'].tolist()
    y = df['label'].astype(int).tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
    return X_train, X_test, y_train, y_test

def train_models(X_train: List[str], y_train: List[int]):
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    # RandomForest
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    rf_model.fit(X_train_tfidf, y_train)
    # XGBoost
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=400, learning_rate=0.05, max_depth=6)
    xgb_model.fit(X_train_tfidf, y_train)
    # Return all + vectorizer
    return {'random_forest': rf_model, 'xgboost': xgb_model}, vectorizer

def evaluate_model(model, vectorizer, X_test: List[str], y_test: List[int]) -> dict:
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)
    # If predict_proba missing (e.g., LinearSVC), we adapt probability later; both our models have proba or we use majority voting.
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "confusion_matrix": cm}

def select_and_save_best(models: dict, vectorizer, X_test, y_test):
    results = {}
    best_name = None
    best_f1 = -1
    for name, m in models.items():
        metrics = evaluate_model(m, vectorizer, X_test, y_test)
        results[name] = metrics
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_name = name
    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(vectorizer, 'artifacts/tfidf_vectorizer.pkl')
    joblib.dump(models[best_name], 'artifacts/best_model.pkl')
    with open('artifacts/metrics.json','w') as f:
        import json; json.dump(results,f,indent=2)
    print('\nModel comparison:')
    for k,v in results.items():
        print(f"{k}: F1={v['f1']:.3f} Acc={v['accuracy']:.3f} P={v['precision']:.3f} R={v['recall']:.3f}")
    print(f"\nBest model: {best_name} (F1={best_f1:.3f}) saved to artifacts/best_model.pkl")

# Placeholder XAI explanation
# For real SHAP/LIME, integrate those libraries here.
def get_xai_explanation(text: str) -> str:
    return " ".join([w.upper() if len(w)>6 else w for w in text.split()])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='dataset.csv', help='CSV with text,label columns')
    args = ap.parse_args()
    X_train, X_test, y_train, y_test = load_and_split_data(args.data)
    models, vectorizer = train_models(X_train, y_train)
    select_and_save_best(models, vectorizer, X_test, y_test)
    print("Training complete and artifacts saved in ./artifacts")

if __name__ == '__main__':
    main()
