"""
Retrain Logistic Regression with Better Class Balance and Threshold Tuning
Focus: Reduce false negatives (catching more misinformation)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_recall_curve
import joblib
import os
import json
from datetime import datetime

print("=" * 70)
print("RETRAINING WITH BALANCED CLASSES & THRESHOLD TUNING")
print("=" * 70)

# Load datasets
print("\n[1/7] Loading datasets...")
misinfo_df = pd.read_csv('data/misinfo_train.csv')
nonmisinfo_df = pd.read_csv('data/nonmisinfo_train.csv')

print(f"  ✓ Misinformation samples: {len(misinfo_df)}")
print(f"  ✓ Non-misinformation samples: {len(nonmisinfo_df)}")

# Use 1:2 ratio for better recall (instead of 1:5)
misinfo_count = len(misinfo_df)
nonmisinfo_sample = nonmisinfo_df.sample(n=min(misinfo_count * 2, len(nonmisinfo_df)), random_state=42)

print(f"\n[2/7] Creating balanced dataset (1:2 ratio for better misinfo detection)...")
print(f"  ✓ Using {misinfo_count} misinformation samples")
print(f"  ✓ Using {len(nonmisinfo_sample)} non-misinformation samples")

# Combine datasets
misinfo_df['label'] = 1
nonmisinfo_sample['label'] = 0
combined_df = pd.concat([misinfo_df, nonmisinfo_sample], ignore_index=True)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

X = combined_df['text'].fillna('')
y = combined_df['label']

print(f"  ✓ Total samples: {len(combined_df)}")
print(f"  ✓ Class distribution: {y.value_counts().to_dict()}")

# Split for threshold tuning
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(f"\n[3/7] Split: {len(X_train)} train, {len(X_val)} validation")

# Vectorization
print("\n[4/7] Vectorizing text...")
vectorizer = TfidfVectorizer(
    max_features=7500,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.85,
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
print(f"  ✓ Feature count: {X_train_vec.shape[1]}")

# Train model with strong class weighting
print("\n[5/7] Training Logistic Regression...")
# Use higher weight for misinformation class
model = LogisticRegression(
    C=1.0,
    penalty='l2',
    solver='saga',
    max_iter=2000,
    random_state=42,
    class_weight={0: 1, 1: 3},  # 3x weight for misinformation
    tol=0.0001
)

model.fit(X_train_vec, y_train)
print("  ✓ Model trained")

# Find optimal threshold on validation set
print("\n[6/7] Finding optimal decision threshold...")
y_proba = model.predict_proba(X_val_vec)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)

# Calculate F1 scores for different thresholds
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

# Find threshold that maximizes F1 score
best_threshold_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else 0.5

print(f"  ✓ Optimal threshold: {best_threshold:.3f} (default: 0.500)")
print(f"  ✓ Best F1 score: {f1_scores[best_threshold_idx]:.3f}")

# Evaluate with optimal threshold
y_pred_optimal = (y_proba >= best_threshold).astype(int)

accuracy_optimal = accuracy_score(y_val, y_pred_optimal)
f1_optimal = f1_score(y_val, y_pred_optimal)

print(f"\n  Performance with optimal threshold:")
print(f"    Accuracy: {accuracy_optimal*100:.2f}%")
print(f"    F1 Score: {f1_optimal*100:.2f}%")

# Retrain on full dataset
print("\n[7/7] Retraining on full dataset...")
X_full_vec = vectorizer.fit_transform(X)
model.fit(X_full_vec, y)

# Test on full data with optimal threshold
y_proba_full = model.predict_proba(X_full_vec)[:, 1]
y_pred_full = (y_proba_full >= best_threshold).astype(int)

accuracy = accuracy_score(y, y_pred_full)
f1 = f1_score(y, y_pred_full)

print("\n" + "=" * 70)
print("FINAL MODEL PERFORMANCE")
print("=" * 70)
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"F1 Score: {f1*100:.2f}%")
print(f"Decision Threshold: {best_threshold:.3f}")
print("\nClassification Report:")
print(classification_report(y, y_pred_full, target_names=['Non-Misinformation', 'Misinformation']))
print("\nConfusion Matrix:")
cm = confusion_matrix(y, y_pred_full)
print(cm)
print(f"\nTrue Negatives: {cm[0][0]}, False Positives: {cm[0][1]}")
print(f"False Negatives: {cm[1][0]}, True Positives: {cm[1][1]}")

# Test on known examples
print("\n" + "=" * 70)
print("TESTING ON KNOWN MISINFORMATION EXAMPLES")
print("=" * 70)

test_misinfo = [
    "COVID-19 vaccines contain microchips to track people",
    "Bill Gates is using 5G towers to spread coronavirus",
    "Drinking bleach cures cancer and COVID-19",
    "The earth is flat and NASA is lying to everyone",
    "Vaccines cause autism in children"
]

test_nonmisinfo = [
    "The World Health Organization recommends wearing masks during the pandemic",
    "Studies show regular exercise improves cardiovascular health",
    "Climate scientists report rising global temperatures"
]

print("\nMisinformation examples:")
for text in test_misinfo:
    X_test = vectorizer.transform([text])
    proba = model.predict_proba(X_test)[0][1]
    pred = 1 if proba >= best_threshold else 0
    status = "✓" if pred == 1 else "✗"
    print(f"{status} '{text[:50]}...' -> Prob={proba:.2f}, Pred={'MISINFO' if pred==1 else 'NON-MISINFO'}")

print("\nNon-misinformation examples:")
for text in test_nonmisinfo:
    X_test = vectorizer.transform([text])
    proba = model.predict_proba(X_test)[0][1]
    pred = 1 if proba >= best_threshold else 0
    status = "✓" if pred == 0 else "✗"
    print(f"{status} '{text[:50]}...' -> Prob={proba:.2f}, Pred={'MISINFO' if pred==1 else 'NON-MISINFO'}")

# Save model, vectorizer, and threshold
print("\n" + "=" * 70)
print("SAVING MODEL")
print("=" * 70)

os.makedirs('models', exist_ok=True)

model_path = 'models/logistic.pkl'
vectorizer_path = 'models/logistic_tfidf.pkl'
threshold_path = 'models/logistic_threshold.pkl'

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)
joblib.dump(best_threshold, threshold_path)

print(f"✓ Model saved: {model_path}")
print(f"✓ Vectorizer saved: {vectorizer_path}")
print(f"✓ Threshold saved: {threshold_path}")

# Update model summary
summary_path = 'models/model_summary.json'
summary = {
    "best_model": "Logistic Regression (Balanced + Threshold Tuned)",
    "best_model_file": "logistic.pkl",
    "vectorizer_file": "logistic_tfidf.pkl",
    "threshold_file": "logistic_threshold.pkl",
    "decision_threshold": float(best_threshold),
    "accuracy": float(accuracy),
    "f1_score": float(f1),
    "precision": float(classification_report(y, y_pred_full, output_dict=True)['1']['precision']),
    "recall": float(classification_report(y, y_pred_full, output_dict=True)['1']['recall']),
    "training_samples": len(combined_df),
    "class_weight": {0: 1, 1: 3},
    "trained_at": datetime.now().isoformat()
}

with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"✓ Summary saved: {summary_path}")

# Update model config
config_path = 'models/model_config.json'
config = {
    "active_model": {
        "id": "logistic",
        "name": "Logistic Regression (Balanced + Threshold Tuned)",
        "type": "ml",
        "model_file": "logistic.pkl",
        "vectorizer_file": "logistic_tfidf.pkl",
        "threshold_file": "logistic_threshold.pkl",
        "available": True,
        "accuracy": float(accuracy),
        "f1": float(f1),
        "precision": float(classification_report(y, y_pred_full, output_dict=True)['1']['precision']),
        "recall": float(classification_report(y, y_pred_full, output_dict=True)['1']['recall']),
        "decision_threshold": float(best_threshold)
    },
    "last_updated": datetime.now().isoformat(),
    "updated_by": "balanced_retraining"
}

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"✓ Config updated: {config_path}")

print("\n" + "=" * 70)
print("✓ RETRAINING COMPLETE!")
print("=" * 70)
print(f"\nKey Improvements:")
print(f"  • Used 1:2 class ratio (more balanced)")
print(f"  • Applied 3x weight to misinformation class")
print(f"  • Tuned decision threshold to {best_threshold:.3f}")
print(f"  • Accuracy: {accuracy*100:.2f}%")
print(f"  • F1 Score: {f1*100:.2f}%")
print(f"\nRestart the Flask app to use the new model.")
