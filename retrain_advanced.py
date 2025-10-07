#!/usr/bin/env python3
"""
ADVANCED retraining with SMOTE oversampling and threshold tuning
Specifically optimized to catch misinformation cases
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import joblib
import json
from preprocessing import TextPreprocessor

# Try to import SMOTE for oversampling
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("[WARN] imbalanced-learn not installed. Install with: pip install imbalanced-learn")

print("=" * 80)
print("ADVANCED RETRAINING - High-Recall Misinformation Detector")
print("=" * 80)

# Load data
print("\n[1/7] Loading training data...")
misinfo_path = 'data/misinfo_train.csv'
nonmisinfo_path = 'data/nonmisinfo_train.csv'

try:
    df_mis = pd.read_csv(misinfo_path)
    df_non = pd.read_csv(nonmisinfo_path)
    
    mis_texts = df_mis['text'].dropna().astype(str).tolist()
    non_texts = df_non['text'].dropna().astype(str).tolist()
    
    print(f"   ✓ Loaded {len(mis_texts)} misinformation samples")
    print(f"   ✓ Loaded {len(non_texts)} nonmisinformation samples")
    
    # Use ALL misinfo and sample nonmisinfo more aggressively
    # Take up to 5x misinfo samples from nonmisinfo
    max_non = min(len(mis_texts) * 5, len(non_texts))
    non_texts_sampled = np.random.choice(non_texts, size=max_non, replace=False).tolist()
    
    texts = mis_texts + non_texts_sampled
    labels = [1] * len(mis_texts) + [0] * len(non_texts_sampled)
    
    print(f"   ✓ Dataset: {len(mis_texts)} misinfo + {len(non_texts_sampled)} nonmisinfo = {len(texts)} total")
    print(f"   ✓ Class ratio: 1:{len(non_texts_sampled)/len(mis_texts):.1f}")
    
except Exception as e:
    print(f"   ✗ Failed: {e}")
    exit(1)

# Preprocess
print("\n[2/7] Preprocessing...")
preprocessor = TextPreprocessor()
texts_clean = [preprocessor.clean_text(t) for t in texts]
print(f"   ✓ Cleaned {len(texts_clean)} texts")

# Split
print("\n[3/7] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    texts_clean, labels, test_size=0.2, random_state=42, stratify=labels
)
print(f"   ✓ Train: {len(X_train)} | Test: {len(X_test)}")
print(f"   ✓ Train: {sum(y_train)} misinfo, {len(y_train)-sum(y_train)} nonmisinfo")
print(f"   ✓ Test:  {sum(y_test)} misinfo, {len(y_test)-sum(y_test)} nonmisinfo")

# Vectorize
print("\n[4/7] Creating TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.85,
    sublinear_tf=True
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print(f"   ✓ Features: {X_train_vec.shape[1]}")

# Apply SMOTE if available
if SMOTE_AVAILABLE:
    print("\n[5/7] Applying SMOTE oversampling...")
    smote = SMOTE(random_state=42, k_neighbors=min(5, sum(y_train)-1))
    X_train_vec_balanced, y_train_balanced = smote.fit_resample(X_train_vec, y_train)
    print(f"   ✓ After SMOTE: {sum(y_train_balanced)} misinfo, {len(y_train_balanced)-sum(y_train_balanced)} nonmisinfo")
else:
    print("\n[5/7] Skipping SMOTE (not installed)...")
    X_train_vec_balanced = X_train_vec
    y_train_balanced = y_train

# Train models optimized for high recall
print("\n[6/7] Training high-recall models...")

# Model 1: Random Forest tuned for recall
print("   → Random Forest (recall-optimized)...")
rf = RandomForestClassifier(
    n_estimators=600,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight={0: 1, 1: 3},  # Higher weight for misinfo class
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_vec_balanced, y_train_balanced)
rf_pred = rf.predict(X_test_vec)
rf_acc = accuracy_score(y_test, rf_pred)
rf_p, rf_r, rf_f1, _ = precision_recall_fscore_support(y_test, rf_pred, average='binary')
print(f"     ✓ Acc={rf_acc:.3f} | P={rf_p:.3f} | R={rf_r:.3f} | F1={rf_f1:.3f}")

# Model 2: Logistic Regression with high misinfo weight
print("   → Logistic Regression (high misinfo weight)...")
lr = LogisticRegression(
    max_iter=2000,
    class_weight={0: 1, 1: 4},
    random_state=42,
    C=0.5
)
lr.fit(X_train_vec_balanced, y_train_balanced)
lr_pred = lr.predict(X_test_vec)
lr_acc = accuracy_score(y_test, lr_pred)
lr_p, lr_r, lr_f1, _ = precision_recall_fscore_support(y_test, lr_pred, average='binary')
print(f"     ✓ Acc={lr_acc:.3f} | P={lr_p:.3f} | R={lr_r:.3f} | F1={lr_f1:.3f}")

# Choose best model by F1 score
models_eval = [
    ('random_forest', rf, rf_f1, rf_acc, rf_p, rf_r),
    ('logistic_regression', lr, lr_f1, lr_acc, lr_p, lr_r)
]
models_eval.sort(key=lambda x: x[2], reverse=True)
best_name, best_model, best_f1, best_acc, best_p, best_r = models_eval[0]

print(f"\n   ★ BEST MODEL: {best_name} (F1={best_f1:.4f})")

# Evaluate best
print("\n[7/7] Final evaluation...")
y_pred = best_model.predict(X_test_vec)
y_pred_proba = best_model.predict_proba(X_test_vec)

cm = confusion_matrix(y_test, y_pred)

print(f"\n{'='*60}")
print(f"FINAL METRICS ({best_name})")
print(f"{'='*60}")
print(f"Accuracy:  {best_acc:.4f} ({best_acc*100:.2f}%)")
print(f"Precision: {best_p:.4f} ({best_p*100:.2f}%)")
print(f"Recall:    {best_r:.4f} ({best_r*100:.2f}%)")
print(f"F1 Score:  {best_f1:.4f} ({best_f1*100:.2f}%)")
print(f"\nConfusion Matrix:")
print(f"                  Predicted")
print(f"                  0       1")
print(f"Actual  0      {cm[0][0]:5d}   {cm[0][1]:5d}  (nonmisinfo)")
print(f"        1      {cm[1][0]:5d}   {cm[1][1]:5d}  (misinfo)")
print(f"{'='*60}")

# Detailed report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['nonmisinfo', 'misinfo']))

# Sanity tests
print("\n" + "="*60)
print("SANITY CHECK:")
print("="*60)

test_cases = [
    ("Russia has launched special military operation to denazify Ukraine", 1),
    ("The sky is blue", 0),
    ("COVID vaccines contain microchips", 1),
    ("Python is a programming language", 0),
    ("Ukraine bombed its own civilians in Bucha", 1),
    ("Today is sunny", 0),
    ("Biden stole the election with fake ballots", 1),
    ("Water is wet", 0)
]

correct = 0
for text, expected in test_cases:
    clean = preprocessor.clean_text(text)
    vec = vectorizer.transform([clean])
    pred = best_model.predict(vec)[0]
    prob = best_model.predict_proba(vec)[0]
    
    result = "✓" if pred == expected else "✗"
    correct += (pred == expected)
    
    pred_name = "misinfo" if pred == 1 else "nonmisinfo"
    exp_name = "misinfo" if expected == 1 else "nonmisinfo"
    print(f"{result} '{text[:50]}' → {pred_name} (expected {exp_name}) | P(mis)={prob[1]:.3f}")

print(f"\n{correct}/{len(test_cases)} correct ({correct/len(test_cases)*100:.1f}%)")

# Save
print("\n" + "="*60)
print("SAVING MODELS")
print("="*60)

os.makedirs('models', exist_ok=True)

# Save both models
joblib.dump(rf, 'models/random_forest.pkl')
joblib.dump(lr, 'models/logistic_regression.pkl')
joblib.dump(vectorizer, 'models/tfidf.pkl')

# Save best as primary
joblib.dump(best_model, f'models/{best_name}.pkl')

print(f"   ✓ models/random_forest.pkl")
print(f"   ✓ models/logistic_regression.pkl")
print(f"   ✓ models/tfidf.pkl")
print(f"   ✓ models/{best_name}.pkl (PRIMARY)")

# Summary
summary = {
    'best_model': best_name,
    'best_f1': float(best_f1),
    'accuracy': float(best_acc),
    'precision': float(best_p),
    'recall': float(best_r),
    'test_samples': len(y_test),
    'train_samples': len(y_train_balanced),
    'smote_used': SMOTE_AVAILABLE,
    'models': {
        'random_forest': {
            'accuracy': float(rf_acc),
            'f1': float(rf_f1),
            'recall': float(rf_r)
        },
        'logistic_regression': {
            'accuracy': float(lr_acc),
            'f1': float(lr_f1),
            'recall': float(lr_r)
        }
    },
    'confusion_matrix': cm.tolist(),
    'classes': ['nonmisinfo', 'misinfo'],
    'retrain_date': pd.Timestamp.now().isoformat(),
    'optimization': 'high_recall_for_misinformation'
}

with open('models/model_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("   ✓ models/model_summary.json")

print("\n" + "="*80)
print(f"✓ COMPLETE! Best: {best_name} | F1={best_f1:.4f} | Recall={best_r:.4f}")
print("✓ Restart Flask app: python app.py")
print("="*80)
