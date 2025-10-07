#!/usr/bin/env python3
"""
ULTIMATE TRAINING SYSTEM
- Hyperparameter tuning for all ML models
- Pre-trained transformer models (BERT, RoBERTa, DistilBERT, etc.)
- Neural networks (LSTM, BiLSTM, CNN)
- Comprehensive ranking table
- Automatic best model selection
"""
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report, make_scorer, f1_score
import joblib
import json
from preprocessing import TextPreprocessor

# Try to import advanced libraries
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARN] XGBoost not available. Install: pip install xgboost")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] PyTorch not available. Install: pip install torch")

try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        Trainer, TrainingArguments, EarlyStoppingCallback
    )
    from datasets import Dataset as HFDataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[WARN] Transformers not available. Install: pip install transformers datasets")

print("=" * 100)
print(" " * 30 + "🚀 ULTIMATE MISINFORMATION DETECTOR TRAINING 🚀")
print("=" * 100)

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================
print("\n" + "=" * 100)
print("PHASE 1: DATA LOADING & PREPROCESSING")
print("=" * 100)

misinfo_path = 'data/misinfo_train.csv'
nonmisinfo_path = 'data/nonmisinfo_train.csv'

try:
    df_mis = pd.read_csv(misinfo_path, low_memory=False)
    df_non = pd.read_csv(nonmisinfo_path, low_memory=False)
    
    mis_texts = df_mis['text'].dropna().astype(str).tolist()
    non_texts = df_non['text'].dropna().astype(str).tolist()
    
    print(f"✓ Loaded {len(mis_texts)} misinformation samples")
    print(f"✓ Loaded {len(non_texts)} nonmisinformation samples")
    
    # Balance dataset (take all misinfo, sample 3x from nonmisinfo)
    max_non = min(len(mis_texts) * 3, len(non_texts))
    non_texts_sampled = np.random.choice(non_texts, size=max_non, replace=False).tolist()
    
    texts = mis_texts + non_texts_sampled
    labels = [1] * len(mis_texts) + [0] * len(non_texts_sampled)
    
    print(f"✓ Balanced dataset: {len(mis_texts)} misinfo + {len(non_texts_sampled)} nonmisinfo = {len(texts)} total")
    
except Exception as e:
    print(f"✗ Failed to load data: {e}")
    sys.exit(1)

# Preprocess
print("\nPreprocessing texts...")
preprocessor = TextPreprocessor()
texts_clean = [preprocessor.clean_text(t) for t in texts]
print(f"✓ Cleaned {len(texts_clean)} texts")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    texts_clean, labels, test_size=0.2, random_state=42, stratify=labels
)
print(f"✓ Train: {len(X_train)} samples ({sum(y_train)} misinfo, {len(y_train)-sum(y_train)} nonmisinfo)")
print(f"✓ Test:  {len(X_test)} samples ({sum(y_test)} misinfo, {len(y_test)-sum(y_test)} nonmisinfo)")

# Vectorize for classical ML
print("\nCreating TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=12000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.85,
    sublinear_tf=True
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print(f"✓ Feature dimensions: {X_train_vec.shape[1]}")

# ============================================================================
# STORE RESULTS
# ============================================================================
results = []

def evaluate_model(name, model, X_test_vec, y_test, train_time):
    """Evaluate model and store results"""
    y_pred = model.predict(X_test_vec)
    
    acc = accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    results.append({
        'name': name,
        'model': model,
        'accuracy': acc,
        'precision': p,
        'recall': r,
        'f1': f1,
        'train_time': train_time,
        'confusion_matrix': cm
    })
    
    return acc, p, r, f1, cm

# ============================================================================
# PHASE 2: HYPERPARAMETER-TUNED MACHINE LEARNING MODELS
# ============================================================================
print("\n" + "=" * 100)
print("PHASE 2: HYPERPARAMETER-TUNED MACHINE LEARNING MODELS")
print("=" * 100)

# 1. LOGISTIC REGRESSION with GridSearch
print("\n[1/5] 🎯 Logistic Regression (Hyperparameter Tuning)...")
start_time = time.time()
lr_params = {
    'C': [0.1, 0.5, 1.0, 2.0, 5.0],
    'class_weight': [{0: 1, 1: 2}, {0: 1, 1: 3}, {0: 1, 1: 4}, 'balanced'],
    'max_iter': [2000],
    'solver': ['lbfgs', 'liblinear']
}
lr_grid = GridSearchCV(
    LogisticRegression(random_state=42),
    lr_params,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=0
)
lr_grid.fit(X_train_vec, y_train)
lr_best = lr_grid.best_estimator_
lr_time = time.time() - start_time

print(f"  ✓ Best params: {lr_grid.best_params_}")
acc, p, r, f1, cm = evaluate_model('Logistic Regression (Tuned)', lr_best, X_test_vec, y_test, lr_time)
print(f"  ✓ Acc: {acc:.4f} | Prec: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f} | Time: {lr_time:.2f}s")

# 2. RANDOM FOREST with GridSearch
print("\n[2/5] 🌲 Random Forest (Hyperparameter Tuning)...")
start_time = time.time()
rf_params = {
    'n_estimators': [300, 500, 700],
    'max_depth': [None, 50, 100],
    'min_samples_split': [2, 5],
    'class_weight': [{0: 1, 1: 2}, {0: 1, 1: 3}, 'balanced']
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_params,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    verbose=0
)
rf_grid.fit(X_train_vec, y_train)
rf_best = rf_grid.best_estimator_
rf_time = time.time() - start_time

print(f"  ✓ Best params: {rf_grid.best_params_}")
acc, p, r, f1, cm = evaluate_model('Random Forest (Tuned)', rf_best, X_test_vec, y_test, rf_time)
print(f"  ✓ Acc: {acc:.4f} | Prec: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f} | Time: {rf_time:.2f}s")

# 3. LINEAR SVM with GridSearch
print("\n[3/5] ⚡ Linear SVM (Hyperparameter Tuning)...")
start_time = time.time()
svm_params = {
    'C': [0.1, 0.5, 1.0, 2.0],
    'class_weight': [{0: 1, 1: 2}, {0: 1, 1: 3}, {0: 1, 1: 4}, 'balanced'],
    'max_iter': [3000]
}
svm_grid = GridSearchCV(
    LinearSVC(random_state=42, dual=False),
    svm_params,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=0
)
svm_grid.fit(X_train_vec, y_train)
svm_best = svm_grid.best_estimator_
svm_time = time.time() - start_time

print(f"  ✓ Best params: {svm_grid.best_params_}")
acc, p, r, f1, cm = evaluate_model('Linear SVM (Tuned)', svm_best, X_test_vec, y_test, svm_time)
print(f"  ✓ Acc: {acc:.4f} | Prec: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f} | Time: {svm_time:.2f}s")

# 4. GRADIENT BOOSTING
print("\n[4/5] 🚀 Gradient Boosting (Tuned)...")
start_time = time.time()
gb_params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}
gb_grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    gb_params,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    verbose=0
)
gb_grid.fit(X_train_vec, y_train)
gb_best = gb_grid.best_estimator_
gb_time = time.time() - start_time

print(f"  ✓ Best params: {gb_grid.best_params_}")
acc, p, r, f1, cm = evaluate_model('Gradient Boosting (Tuned)', gb_best, X_test_vec, y_test, gb_time)
print(f"  ✓ Acc: {acc:.4f} | Prec: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f} | Time: {gb_time:.2f}s")

# 5. XGBOOST (if available)
if XGBOOST_AVAILABLE:
    print("\n[5/5] ⚡ XGBoost (Hyperparameter Tuning)...")
    start_time = time.time()
    
    # Calculate scale_pos_weight for class imbalance
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    
    xgb_params = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'scale_pos_weight': [scale_pos_weight, scale_pos_weight * 1.5, scale_pos_weight * 2]
    }
    xgb_grid = GridSearchCV(
        XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        xgb_params,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    xgb_grid.fit(X_train_vec, y_train)
    xgb_best = xgb_grid.best_estimator_
    xgb_time = time.time() - start_time
    
    print(f"  ✓ Best params: {xgb_grid.best_params_}")
    acc, p, r, f1, cm = evaluate_model('XGBoost (Tuned)', xgb_best, X_test_vec, y_test, xgb_time)
    print(f"  ✓ Acc: {acc:.4f} | Prec: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f} | Time: {xgb_time:.2f}s")

# ============================================================================
# PHASE 3: PRE-TRAINED TRANSFORMER MODELS
# ============================================================================
if TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
    print("\n" + "=" * 100)
    print("PHASE 3: PRE-TRAINED TRANSFORMER MODELS (BERT, RoBERTa, DistilBERT)")
    print("=" * 100)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare dataset for transformers
    train_df = pd.DataFrame({'text': X_train, 'label': y_train})
    test_df = pd.DataFrame({'text': X_test, 'label': y_test})
    
    train_dataset = HFDataset.from_pandas(train_df)
    test_dataset = HFDataset.from_pandas(test_df)
    
    transformer_models = [
        ('distilbert-base-uncased', 'DistilBERT'),
        ('bert-base-uncased', 'BERT-base'),
        ('roberta-base', 'RoBERTa-base'),
    ]
    
    for model_name, display_name in transformer_models:
        print(f"\n🤖 Training {display_name}...")
        start_time = time.time()
        
        try:
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=2,
                ignore_mismatched_sizes=True
            )
            
            # Tokenize datasets
            def tokenize_function(examples):
                return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)
            
            tokenized_train = train_dataset.map(tokenize_function, batched=True)
            tokenized_test = test_dataset.map(tokenize_function, batched=True)
            
            # Training arguments (fast training)
            training_args = TrainingArguments(
                output_dir=f'./models/transformer_{model_name.split("/")[-1]}',
                evaluation_strategy='epoch',
                save_strategy='epoch',
                learning_rate=2e-5,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                num_train_epochs=3,
                weight_decay=0.01,
                load_best_model_at_end=True,
                metric_for_best_model='f1',
                logging_steps=50,
                save_total_limit=1,
                fp16=torch.cuda.is_available()
            )
            
            # Compute metrics
            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                predictions = predictions.argmax(axis=1)
                acc = accuracy_score(labels, predictions)
                p, r, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)
                return {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1}
            
            # Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_test,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
            )
            
            # Train
            trainer.train()
            
            # Evaluate
            eval_results = trainer.evaluate()
            train_time_transformer = time.time() - start_time
            
            print(f"  ✓ {display_name} trained in {train_time_transformer:.2f}s")
            print(f"  ✓ Acc: {eval_results['eval_accuracy']:.4f} | Prec: {eval_results['eval_precision']:.4f} | Recall: {eval_results['eval_recall']:.4f} | F1: {eval_results['eval_f1']:.4f}")
            
            # Get predictions for confusion matrix
            predictions = trainer.predict(tokenized_test)
            y_pred = predictions.predictions.argmax(axis=1)
            cm = confusion_matrix(y_test, y_pred)
            
            results.append({
                'name': display_name,
                'model': None,  # Don't store transformer model (too large)
                'model_path': training_args.output_dir,
                'accuracy': eval_results['eval_accuracy'],
                'precision': eval_results['eval_precision'],
                'recall': eval_results['eval_recall'],
                'f1': eval_results['eval_f1'],
                'train_time': train_time_transformer,
                'confusion_matrix': cm
            })
            
        except Exception as e:
            print(f"  ✗ Failed to train {display_name}: {e}")

# ============================================================================
# FINAL RANKING TABLE
# ============================================================================
print("\n" + "=" * 100)
print(" " * 40 + "🏆 FINAL MODEL RANKING 🏆")
print("=" * 100)

# Sort by F1 score
results_sorted = sorted(results, key=lambda x: x['f1'], reverse=True)

print(f"\n{'Rank':<6} {'Model':<35} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'Time (s)':<10}")
print("-" * 100)

for i, res in enumerate(results_sorted, 1):
    emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
    print(f"{emoji} {i:<4} {res['name']:<35} {res['accuracy']*100:>10.2f}% {res['precision']*100:>10.2f}% {res['recall']*100:>10.2f}% {res['f1']*100:>10.2f}% {res['train_time']:>9.1f}")

# ============================================================================
# SAVE BEST MODEL
# ============================================================================
best_result = results_sorted[0]
print("\n" + "=" * 100)
print(f"🏆 BEST MODEL: {best_result['name']}")
print(f"   F1 Score: {best_result['f1']:.4f} ({best_result['f1']*100:.2f}%)")
print(f"   Accuracy: {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)")
print(f"   Precision: {best_result['precision']:.4f} ({best_result['precision']*100:.2f}%)")
print(f"   Recall: {best_result['recall']:.4f} ({best_result['recall']*100:.2f}%)")
print("=" * 100)

# Save models
os.makedirs('models', exist_ok=True)

# Save vectorizer
joblib.dump(vectorizer, 'models/tfidf.pkl')
print("\n✓ Saved: models/tfidf.pkl")

# Save all classical models
model_names_map = {
    'Logistic Regression (Tuned)': ('logistic_regression', lr_best),
    'Random Forest (Tuned)': ('random_forest', rf_best),
    'Linear SVM (Tuned)': ('svm', svm_best),
    'Gradient Boosting (Tuned)': ('gradient_boosting', gb_best),
}

if XGBOOST_AVAILABLE:
    model_names_map['XGBoost (Tuned)'] = ('xgboost', xgb_best)

for res in results:
    if res['name'] in model_names_map and res['model'] is not None:
        file_name, model_obj = model_names_map[res['name']]
        joblib.dump(model_obj, f'models/{file_name}.pkl')
        print(f"✓ Saved: models/{file_name}.pkl")

# Save model summary
summary = {
    'best_model': best_result['name'],
    'best_model_file': model_names_map.get(best_result['name'], ('unknown', None))[0] if best_result['name'] in model_names_map else best_result.get('model_path', 'transformer'),
    'best_f1': float(best_result['f1']),
    'accuracy': float(best_result['accuracy']),
    'precision': float(best_result['precision']),
    'recall': float(best_result['recall']),
    'training_date': pd.Timestamp.now().isoformat(),
    'all_models': [
        {
            'rank': i+1,
            'name': res['name'],
            'accuracy': float(res['accuracy']),
            'precision': float(res['precision']),
            'recall': float(res['recall']),
            'f1': float(res['f1']),
            'train_time': float(res['train_time']),
            'confusion_matrix': res['confusion_matrix'].tolist()
        }
        for i, res in enumerate(results_sorted)
    ]
}

with open('models/model_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("✓ Saved: models/model_summary.json")

print("\n" + "=" * 100)
print("✅ TRAINING COMPLETE!")
print(f"✅ Best model ready: {best_result['name']}")
print("✅ Restart Flask app to use the new model: python app.py")
print("=" * 100)
