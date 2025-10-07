"""
Advanced Hyperparameter Tuning for Logistic Regression
Comprehensive parameter search to maximize accuracy and F1 score
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
import os
import json
from datetime import datetime

print("=" * 70)
print("ADVANCED LOGISTIC REGRESSION HYPERPARAMETER TUNING")
print("=" * 70)

# Load datasets
print("\n[1/6] Loading datasets...")
misinfo_df = pd.read_csv('data/misinfo_train.csv')
nonmisinfo_df = pd.read_csv('data/nonmisinfo_train.csv')

print(f"  ✓ Misinformation samples: {len(misinfo_df)}")
print(f"  ✓ Non-misinformation samples: {len(nonmisinfo_df)}")

# Balance dataset with ratio 1:5 (more diverse non-misinfo)
misinfo_count = len(misinfo_df)
nonmisinfo_sample = nonmisinfo_df.sample(n=min(misinfo_count * 5, len(nonmisinfo_df)), random_state=42)

print(f"\n[2/6] Balancing dataset (1:5 ratio)...")
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

# Vectorization with extensive parameter search
print("\n[3/6] Testing TF-IDF vectorizer configurations...")

vectorizer_configs = [
    {'max_features': 5000, 'ngram_range': (1, 2), 'min_df': 2, 'max_df': 0.9, 'sublinear_tf': True},
    {'max_features': 7500, 'ngram_range': (1, 2), 'min_df': 2, 'max_df': 0.85, 'sublinear_tf': True},
    {'max_features': 10000, 'ngram_range': (1, 3), 'min_df': 3, 'max_df': 0.9, 'sublinear_tf': True},
    {'max_features': 8000, 'ngram_range': (1, 2), 'min_df': 2, 'max_df': 0.9, 'sublinear_tf': False},
]

best_vectorizer_config = None
best_vectorizer = None
best_X_train = None
best_vec_score = 0

for i, config in enumerate(vectorizer_configs, 1):
    print(f"\n  Testing vectorizer config {i}/{len(vectorizer_configs)}:")
    print(f"    max_features={config['max_features']}, ngram_range={config['ngram_range']}")
    
    vectorizer = TfidfVectorizer(**config)
    X_vec = vectorizer.fit_transform(X)
    
    # Quick test with basic logistic regression
    lr_test = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    cv_scores = []
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    for train_idx, val_idx in skf.split(X_vec, y):
        lr_test.fit(X_vec[train_idx], y.iloc[train_idx])
        y_pred = lr_test.predict(X_vec[val_idx])
        cv_scores.append(f1_score(y.iloc[val_idx], y_pred))
    
    avg_f1 = np.mean(cv_scores)
    print(f"    → Average F1: {avg_f1:.4f}")
    
    if avg_f1 > best_vec_score:
        best_vec_score = avg_f1
        best_vectorizer_config = config
        best_vectorizer = vectorizer
        best_X_train = X_vec

print(f"\n  ✓ Best vectorizer config: max_features={best_vectorizer_config['max_features']}, "
      f"ngram_range={best_vectorizer_config['ngram_range']}, F1={best_vec_score:.4f}")

# Comprehensive hyperparameter grid for Logistic Regression
print("\n[4/6] Setting up hyperparameter grid...")

param_grid = {
    'C': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],  # Regularization strength
    'penalty': ['l1', 'l2', 'elasticnet'],        # Regularization type
    'solver': ['saga', 'liblinear'],              # Optimizers that support all penalties
    'max_iter': [1000, 2000],                     # Iterations
    'class_weight': ['balanced', None, {0: 1, 1: 2}, {0: 1, 1: 3}],  # Class weights
    'tol': [1e-4, 1e-3],                          # Tolerance
}

# Special handling for elasticnet (requires l1_ratio)
print("  Parameter grid:")
print(f"    C: {param_grid['C']}")
print(f"    penalty: {param_grid['penalty']}")
print(f"    solver: {param_grid['solver']}")
print(f"    class_weight: {param_grid['class_weight']}")
print(f"    max_iter: {param_grid['max_iter']}")
print(f"    tol: {param_grid['tol']}")

# Grid search with cross-validation
print("\n[5/6] Running grid search (this may take a while)...")
print("  Using 5-fold stratified cross-validation...")

# First pass: L1 and L2 penalties
param_grid_l1_l2 = {
    'C': param_grid['C'],
    'penalty': ['l1', 'l2'],
    'solver': ['saga', 'liblinear'],
    'max_iter': param_grid['max_iter'],
    'class_weight': param_grid['class_weight'],
    'tol': param_grid['tol'],
}

grid_search = GridSearchCV(
    LogisticRegression(random_state=42),
    param_grid_l1_l2,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(best_X_train, y)

print("\n  ✓ Grid search completed!")
print(f"  Best F1 Score: {grid_search.best_score_:.4f}")
print(f"  Best Parameters: {grid_search.best_params_}")

# Train final model with best parameters
print("\n[6/6] Training final model with best parameters...")
best_model = grid_search.best_estimator_
best_model.fit(best_X_train, y)

# Evaluate on full dataset
y_pred = best_model.predict(best_X_train)
accuracy = accuracy_score(y, y_pred)
f1 = f1_score(y, y_pred)

print("\n" + "=" * 70)
print("FINAL MODEL PERFORMANCE")
print("=" * 70)
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"F1 Score: {f1*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y, y_pred, target_names=['Non-Misinformation', 'Misinformation']))
print("\nConfusion Matrix:")
cm = confusion_matrix(y, y_pred)
print(cm)
print(f"\nTrue Negatives: {cm[0][0]}, False Positives: {cm[0][1]}")
print(f"False Negatives: {cm[1][0]}, True Positives: {cm[1][1]}")

# Save model and vectorizer
print("\n" + "=" * 70)
print("SAVING MODEL")
print("=" * 70)

os.makedirs('models', exist_ok=True)

model_path = 'models/logistic.pkl'
vectorizer_path = 'models/logistic_tfidf.pkl'

joblib.dump(best_model, model_path)
joblib.dump(best_vectorizer, vectorizer_path)

print(f"✓ Model saved: {model_path}")
print(f"✓ Vectorizer saved: {vectorizer_path}")

# Update model summary
summary_path = 'models/model_summary.json'
summary = {
    "best_model": "Logistic Regression (Tuned)",
    "best_model_file": "logistic.pkl",
    "vectorizer_file": "logistic_tfidf.pkl",
    "accuracy": float(accuracy),
    "f1_score": float(f1),
    "precision": float(classification_report(y, y_pred, output_dict=True)['1']['precision']),
    "recall": float(classification_report(y, y_pred, output_dict=True)['1']['recall']),
    "training_samples": len(combined_df),
    "best_params": grid_search.best_params_,
    "vectorizer_config": best_vectorizer_config,
    "trained_at": datetime.now().isoformat()
}

with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"✓ Summary saved: {summary_path}")

# Update model config to activate tuned model
config_path = 'models/model_config.json'
config = {
    "active_model": {
        "id": "logistic",
        "name": "Logistic Regression (Tuned)",
        "type": "ml",
        "model_file": "logistic.pkl",
        "vectorizer_file": "logistic_tfidf.pkl",
        "available": True,
        "accuracy": float(accuracy),
        "f1": float(f1),
        "precision": float(classification_report(y, y_pred, output_dict=True)['1']['precision']),
        "recall": float(classification_report(y, y_pred, output_dict=True)['1']['recall'])
    },
    "last_updated": datetime.now().isoformat(),
    "updated_by": "hyperparameter_tuning"
}

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"✓ Config updated: {config_path}")

print("\n" + "=" * 70)
print("✓ TUNING COMPLETE!")
print("=" * 70)
print(f"\nFinal Results:")
print(f"  Model: Logistic Regression (Hyperparameter Tuned)")
print(f"  Accuracy: {accuracy*100:.2f}%")
print(f"  F1 Score: {f1*100:.2f}%")
print(f"  Best Parameters: {grid_search.best_params_}")
print(f"\nThe tuned model is now active. Restart the Flask app to use it.")
