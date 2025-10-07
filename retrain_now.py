#!/usr/bin/env python3
"""
Emergency retraining script - creates a robust multi-model ensemble
Optimized for accuracy on misinformation vs nonmisinformation classification
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import joblib
import json
from preprocessing import TextPreprocessor

print("=" * 80)
print("EMERGENCY RETRAINING - Building Robust Misinformation Detector")
print("=" * 80)

# Load data
print("\n[1/6] Loading training data...")
misinfo_path = 'data/misinfo_train.csv'
nonmisinfo_path = 'data/nonmisinfo_train.csv'

try:
    df_mis = pd.read_csv(misinfo_path)
    df_non = pd.read_csv(nonmisinfo_path)
    
    # Extract text and label columns
    mis_texts = df_mis['text'].dropna().astype(str).tolist()
    non_texts = df_non['text'].dropna().astype(str).tolist()
    
    print(f"   ✓ Loaded {len(mis_texts)} misinformation samples")
    print(f"   ✓ Loaded {len(non_texts)} nonmisinformation samples")
    
    # Balance dataset (to prevent bias toward majority class)
    min_samples = min(len(mis_texts), len(non_texts))
    # Use all misinfo (minority) and sample from nonmisinfo (majority)
    if len(non_texts) > len(mis_texts) * 3:
        # Take 3x misinfo samples from nonmisinfo for balance
        non_texts_sampled = np.random.choice(non_texts, size=min(len(mis_texts) * 3, len(non_texts)), replace=False).tolist()
    else:
        non_texts_sampled = non_texts
    
    # Combine
    texts = mis_texts + non_texts_sampled
    labels = [1] * len(mis_texts) + [0] * len(non_texts_sampled)
    
    print(f"   ✓ Balanced dataset: {len(mis_texts)} misinfo + {len(non_texts_sampled)} nonmisinfo = {len(texts)} total")
    
except Exception as e:
    print(f"   ✗ Failed to load data: {e}")
    exit(1)

# Preprocess texts
print("\n[2/6] Preprocessing texts...")
preprocessor = TextPreprocessor()
texts_clean = [preprocessor.clean_text(t) for t in texts]
print(f"   ✓ Cleaned {len(texts_clean)} texts")

# Split data
print("\n[3/6] Splitting train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    texts_clean, labels, test_size=0.2, random_state=42, stratify=labels
)
print(f"   ✓ Train: {len(X_train)} samples | Test: {len(X_test)} samples")
print(f"   ✓ Train distribution: {sum(y_train)} misinfo, {len(y_train)-sum(y_train)} nonmisinfo")

# Vectorize
print("\n[4/6] Creating TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 3),  # unigrams, bigrams, trigrams
    min_df=2,
    max_df=0.9,
    sublinear_tf=True
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print(f"   ✓ Feature dimensions: {X_train_vec.shape[1]}")

# Train multiple strong classifiers
print("\n[5/6] Training ensemble of classifiers...")

# Model 1: Random Forest with balanced class weights
print("   → Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_vec, y_train)
rf_score = rf.score(X_test_vec, y_test)
print(f"     ✓ Random Forest accuracy: {rf_score:.4f}")

# Model 2: Logistic Regression with balanced weights
print("   → Training Logistic Regression...")
lr = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42,
    C=1.0
)
lr.fit(X_train_vec, y_train)
lr_score = lr.score(X_test_vec, y_test)
print(f"     ✓ Logistic Regression accuracy: {lr_score:.4f}")

# Model 3: Linear SVM with balanced weights
print("   → Training Linear SVM...")
svm = LinearSVC(
    class_weight='balanced',
    random_state=42,
    max_iter=2000,
    dual=False
)
svm.fit(X_train_vec, y_train)
svm_score = svm.score(X_test_vec, y_test)
print(f"     ✓ Linear SVM accuracy: {svm_score:.4f}")

# Model 4: Gradient Boosting
print("   → Training Gradient Boosting...")
gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gb.fit(X_train_vec, y_train)
gb_score = gb.score(X_test_vec, y_test)
print(f"     ✓ Gradient Boosting accuracy: {gb_score:.4f}")

# Create voting ensemble (soft voting for probability averaging)
print("   → Creating Voting Ensemble...")
ensemble = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('lr', lr),
        ('gb', gb)
    ],
    voting='soft',
    weights=[2, 1, 2]  # Give more weight to RF and GB
)
ensemble.fit(X_train_vec, y_train)
ensemble_score = ensemble.score(X_test_vec, y_test)
print(f"     ✓ Ensemble accuracy: {ensemble_score:.4f}")

# Evaluate best model
print("\n[6/6] Final evaluation...")
y_pred = ensemble.predict(X_test_vec)
y_pred_proba = ensemble.predict_proba(X_test_vec)

acc = accuracy_score(y_test, y_pred)
p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
cm = confusion_matrix(y_test, y_pred)

print(f"\n{'='*60}")
print(f"FINAL METRICS (Voting Ensemble)")
print(f"{'='*60}")
print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
print(f"Precision: {p:.4f} ({p*100:.2f}%)")
print(f"Recall:    {r:.4f} ({r*100:.2f}%)")
print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
print(f"\nConfusion Matrix:")
print(f"                  Predicted")
print(f"                  0       1")
print(f"Actual  0      {cm[0][0]:5d}   {cm[0][1]:5d}  (nonmisinfo)")
print(f"        1      {cm[1][0]:5d}   {cm[1][1]:5d}  (misinfo)")
print(f"{'='*60}")

# Detailed classification report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['nonmisinfo', 'misinfo']))

# Test specific cases
print("\n" + "="*60)
print("SANITY CHECK - Testing known examples:")
print("="*60)

test_cases = [
    ("Russia has launched a special military operation to denazify Ukraine", 1, "misinfo"),
    ("The sky is blue", 0, "nonmisinfo"),
    ("COVID vaccines contain microchips for tracking", 1, "misinfo"),
    ("Python is a programming language", 0, "nonmisinfo"),
    ("Ukraine bombed its own civilians in Bucha", 1, "misinfo"),
    ("Today is a sunny day", 0, "nonmisinfo")
]

correct_predictions = 0
for text, expected_label, expected_name in test_cases:
    clean = preprocessor.clean_text(text)
    vec = vectorizer.transform([clean])
    pred = ensemble.predict(vec)[0]
    prob = ensemble.predict_proba(vec)[0]
    
    result = "✓" if pred == expected_label else "✗"
    correct_predictions += (pred == expected_label)
    
    pred_name = "misinfo" if pred == 1 else "nonmisinfo"
    print(f"{result} '{text[:60]}...'")
    print(f"   Expected: {expected_name} | Predicted: {pred_name} | Probs: [non={prob[0]:.3f}, mis={prob[1]:.3f}]")

print(f"\nSanity check: {correct_predictions}/{len(test_cases)} correct")

# Save models
print("\n" + "="*60)
print("SAVING MODELS")
print("="*60)

os.makedirs('models', exist_ok=True)

# Save individual models for fallback
joblib.dump(rf, 'models/random_forest.pkl')
joblib.dump(lr, 'models/logistic_regression.pkl')
joblib.dump(svm, 'models/svm.pkl')
joblib.dump(gb, 'models/gradient_boosting.pkl')

# Save ensemble as primary model
joblib.dump(ensemble, 'models/ensemble.pkl')
joblib.dump(vectorizer, 'models/tfidf.pkl')

print("   ✓ Saved: models/random_forest.pkl")
print("   ✓ Saved: models/logistic_regression.pkl")
print("   ✓ Saved: models/svm.pkl")
print("   ✓ Saved: models/gradient_boosting.pkl")
print("   ✓ Saved: models/ensemble.pkl (PRIMARY)")
print("   ✓ Saved: models/tfidf.pkl")

# Create model summary
summary = {
    'best_model': 'ensemble',
    'best_f1': float(f1),
    'accuracy': float(acc),
    'precision': float(p),
    'recall': float(r),
    'test_samples': len(y_test),
    'train_samples': len(y_train),
    'models': {
        'random_forest': {'accuracy': float(rf_score)},
        'logistic_regression': {'accuracy': float(lr_score)},
        'svm': {'accuracy': float(svm_score)},
        'gradient_boosting': {'accuracy': float(gb_score)},
        'ensemble': {'accuracy': float(ensemble_score), 'f1': float(f1)}
    },
    'confusion_matrix': cm.tolist(),
    'classes': ['nonmisinfo', 'misinfo'],
    'retrain_date': pd.Timestamp.now().isoformat()
}

with open('models/model_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("   ✓ Saved: models/model_summary.json")

print("\n" + "="*80)
print("✓ RETRAINING COMPLETE!")
print(f"✓ Best Model: Voting Ensemble (F1={f1:.4f}, Accuracy={acc:.4f})")
print("✓ Restart Flask app to load new models")
print("="*80)
