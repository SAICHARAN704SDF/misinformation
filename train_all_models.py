"""
Comprehensive Multi-Model Training System
Trains ML, Neural Network, and LLM models, compares them, and saves the best one.
"""
import os
import json
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import joblib
import warnings
warnings.filterwarnings('ignore')

# Optional: Neural Network and Transformer models
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except:
    TORCH_AVAILABLE = False
    print("[WARN] PyTorch not available - skipping neural network models")

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    from datasets import Dataset as HFDataset
    TRANSFORMERS_AVAILABLE = True
except:
    TRANSFORMERS_AVAILABLE = False
    print("[WARN] Transformers not available - skipping LLM models")

class ModelTrainer:
    """Unified trainer for all model types"""
    
    def __init__(self, misinfo_csv='data/misinfo_train.csv', nonmisinfo_csv='data/nonmisinfo_train.csv'):
        self.misinfo_csv = misinfo_csv
        self.nonmisinfo_csv = nonmisinfo_csv
        self.results = []
        self.best_model = None
        self.best_score = 0
        self.best_model_name = None
        
    def load_data(self, balance=True, max_samples=None):
        """Load and prepare data"""
        print("\n" + "="*70)
        print("LOADING TRAINING DATA")
        print("="*70)
        
        # Load datasets
        df_misinfo = pd.read_csv(self.misinfo_csv)
        df_nonmisinfo = pd.read_csv(self.nonmisinfo_csv)
        
        print(f"✓ Loaded {len(df_misinfo)} misinformation samples")
        print(f"✓ Loaded {len(df_nonmisinfo)} nonmisinformation samples")
        
        # Extract text
        misinfo_texts = df_misinfo['text'].dropna().astype(str).tolist()
        nonmisinfo_texts = df_nonmisinfo['text'].dropna().astype(str).tolist()
        
        # Balance if requested
        if balance:
            min_samples = min(len(misinfo_texts), len(nonmisinfo_texts))
            if max_samples:
                min_samples = min(min_samples, max_samples)
            misinfo_texts = misinfo_texts[:min_samples]
            nonmisinfo_texts = nonmisinfo_texts[:min_samples]
            print(f"✓ Balanced to {min_samples} samples per class")
        elif max_samples:
            misinfo_texts = misinfo_texts[:max_samples]
            nonmisinfo_texts = nonmisinfo_texts[:max_samples]
        
        # Combine
        texts = misinfo_texts + nonmisinfo_texts
        labels = [1] * len(misinfo_texts) + [0] * len(nonmisinfo_texts)
        
        print(f"✓ Total samples: {len(texts)} (misinfo: {sum(labels)}, nonmisinfo: {len(labels)-sum(labels)})")
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def evaluate_model(self, y_true, y_pred, model_name, train_time):
        """Evaluate and display model performance"""
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        
        # Store results
        result = {
            'model': model_name,
            'accuracy': acc,
            'precision': p,
            'recall': r,
            'f1': f1,
            'train_time': train_time,
            'confusion_matrix': cm.tolist()
        }
        self.results.append(result)
        
        # Display
        print("\n" + "="*70)
        print(f"MODEL: {model_name}")
        print("="*70)
        print(f"Training Time: {train_time:.2f}s")
        print(f"\nMetrics:")
        print(f"  Accuracy:  {acc*100:6.2f}%")
        print(f"  Precision: {p*100:6.2f}%")
        print(f"  Recall:    {r*100:6.2f}%")
        print(f"  F1 Score:  {f1*100:6.2f}%")
        print(f"\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"                 Misinfo  Nonmisinfo")
        print(f"  Actual Misinfo    {cm[1][1]:4d}     {cm[1][0]:4d}")
        print(f"  Actual Nonmisinfo {cm[0][1]:4d}     {cm[0][0]:4d}")
        
        # Detailed classification report
        print(f"\nDetailed Report:")
        print(classification_report(y_true, y_pred, target_names=['nonmisinfo', 'misinfo'], zero_division=0))
        
        # Track best model
        if f1 > self.best_score:
            self.best_score = f1
            self.best_model_name = model_name
            print(f"\n🌟 NEW BEST MODEL! (F1: {f1*100:.2f}%)")
        
        return result
    
    def train_random_forest(self, X_train, X_test, y_train, y_test):
        """Train Random Forest classifier"""
        print("\n" + "🌲"*35)
        print("Training: Random Forest")
        
        start = time.time()
        
        # Vectorize
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=50,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_vec, y_train)
        
        train_time = time.time() - start
        
        # Predict
        y_pred = model.predict(X_test_vec)
        
        # Evaluate
        self.evaluate_model(y_test, y_pred, "RandomForest", train_time)
        
        # Save
        joblib.dump(model, 'models/random_forest.pkl')
        joblib.dump(vectorizer, 'models/tfidf.pkl')
        
        return model, vectorizer
    
    def train_xgboost(self, X_train, X_test, y_train, y_test):
        """Train XGBoost classifier"""
        try:
            from xgboost import XGBClassifier
        except:
            print("\n⚠️  XGBoost not installed - skipping")
            return None, None
        
        print("\n" + "🚀"*35)
        print("Training: XGBoost")
        
        start = time.time()
        
        # Vectorize
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Calculate scale_pos_weight
        neg_count = sum(1 for y in y_train if y == 0)
        pos_count = sum(1 for y in y_train if y == 1)
        scale = neg_count / pos_count if pos_count > 0 else 1.0
        
        # Train
        model = XGBClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            scale_pos_weight=scale,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(X_train_vec, y_train)
        
        train_time = time.time() - start
        
        # Predict
        y_pred = model.predict(X_test_vec)
        
        # Evaluate
        self.evaluate_model(y_test, y_pred, "XGBoost", train_time)
        
        # Save
        joblib.dump(model, 'models/xgboost.pkl')
        joblib.dump(vectorizer, 'models/xgboost_tfidf.pkl')
        
        return model, vectorizer
    
    def train_svm(self, X_train, X_test, y_train, y_test):
        """Train SVM classifier"""
        print("\n" + "⚡"*35)
        print("Training: Linear SVM")
        
        start = time.time()
        
        # Vectorize
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train
        model = LinearSVC(
            C=1.0,
            class_weight='balanced',
            random_state=42,
            max_iter=2000
        )
        model.fit(X_train_vec, y_train)
        
        train_time = time.time() - start
        
        # Predict
        y_pred = model.predict(X_test_vec)
        
        # Evaluate
        self.evaluate_model(y_test, y_pred, "LinearSVM", train_time)
        
        # Save
        joblib.dump(model, 'models/svm.pkl')
        joblib.dump(vectorizer, 'models/svm_tfidf.pkl')
        
        return model, vectorizer
    
    def train_logistic_regression(self, X_train, X_test, y_train, y_test):
        """Train Logistic Regression"""
        print("\n" + "📊"*35)
        print("Training: Logistic Regression")
        
        start = time.time()
        
        # Vectorize
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train
        model = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            random_state=42,
            max_iter=1000,
            n_jobs=-1
        )
        model.fit(X_train_vec, y_train)
        
        train_time = time.time() - start
        
        # Predict
        y_pred = model.predict(X_test_vec)
        
        # Evaluate
        self.evaluate_model(y_test, y_pred, "LogisticRegression", train_time)
        
        # Save
        joblib.dump(model, 'models/logistic.pkl')
        joblib.dump(vectorizer, 'models/logistic_tfidf.pkl')
        
        return model, vectorizer
    
    def train_lstm(self, X_train, X_test, y_train, y_test):
        """Train LSTM neural network"""
        if not TORCH_AVAILABLE:
            return None
        
        print("\n" + "🧠"*35)
        print("Training: LSTM Neural Network")
        
        start = time.time()
        
        # Build vocabulary
        from collections import Counter
        all_words = []
        for text in X_train:
            all_words.extend(text.lower().split())
        word_counts = Counter(all_words)
        vocab = {word: idx+2 for idx, (word, _) in enumerate(word_counts.most_common(5000))}
        vocab['<PAD>'] = 0
        vocab['<UNK>'] = 1
        
        # Encode texts
        def encode(text, max_len=100):
            tokens = text.lower().split()[:max_len]
            ids = [vocab.get(w, 1) for w in tokens]
            ids += [0] * (max_len - len(ids))
            return ids
        
        X_train_enc = torch.tensor([encode(x) for x in X_train], dtype=torch.long)
        X_test_enc = torch.tensor([encode(x) for x in X_test], dtype=torch.long)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        y_test_t = torch.tensor(y_test, dtype=torch.long)
        
        # Create dataset
        class TextDataset(Dataset):
            def __init__(self, X, y):
                self.X = X
                self.y = y
            def __len__(self):
                return len(self.y)
            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]
        
        train_dataset = TextDataset(X_train_enc, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Define model
        class LSTMClassifier(nn.Module):
            def __init__(self, vocab_size, embed_dim=128, hidden_dim=128):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
                self.fc = nn.Linear(hidden_dim*2, 2)
                self.dropout = nn.Dropout(0.3)
            
            def forward(self, x):
                emb = self.embedding(x)
                out, _ = self.lstm(emb)
                out = out[:, -1, :]
                out = self.dropout(out)
                return self.fc(out)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LSTMClassifier(len(vocab)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train
        model.train()
        for epoch in range(5):
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"  Epoch {epoch+1}/5, Loss: {total_loss/len(train_loader):.4f}")
        
        train_time = time.time() - start
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            X_test_batch = X_test_enc.to(device)
            outputs = model(X_test_batch)
            y_pred = outputs.argmax(dim=1).cpu().numpy()
        
        self.evaluate_model(y_test, y_pred, "LSTM", train_time)
        
        # Save
        torch.save({
            'model_state': model.state_dict(),
            'vocab': vocab
        }, 'models/rnn_lstm.pt')
        
        return model, vocab
    
    def train_transformer(self, X_train, X_test, y_train, y_test, model_name='distilbert-base-uncased'):
        """Train transformer model"""
        if not TRANSFORMERS_AVAILABLE:
            return None
        
        print("\n" + "🤖"*35)
        print(f"Training: Transformer ({model_name})")
        
        start = time.time()
        
        # Prepare datasets
        train_df = pd.DataFrame({'text': X_train, 'label': y_train})
        test_df = pd.DataFrame({'text': X_test, 'label': y_test})
        
        train_dataset = HFDataset.from_pandas(train_df)
        test_dataset = HFDataset.from_pandas(test_df)
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        
        # Tokenize
        def tokenize(batch):
            return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=128)
        
        train_dataset = train_dataset.map(tokenize, batched=True)
        test_dataset = test_dataset.map(tokenize, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./models/transformer_temp',
            evaluation_strategy='epoch',
            save_strategy='epoch',
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model='f1'
        )
        
        # Compute metrics
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            preds = predictions.argmax(-1)
            p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
            acc = accuracy_score(labels, preds)
            return {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1}
        
        # Train
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
        
        trainer.train()
        train_time = time.time() - start
        
        # Evaluate
        predictions = trainer.predict(test_dataset)
        y_pred = predictions.predictions.argmax(-1)
        
        self.evaluate_model(y_test, y_pred, f"Transformer_{model_name.split('/')[-1]}", train_time)
        
        # Save
        model_slug = model_name.split('/')[-1].replace('-', '_')
        trainer.save_model(f'models/transformer_{model_slug}')
        tokenizer.save_pretrained(f'models/tokenizer_{model_slug}')
        
        return model, tokenizer
    
    def display_final_summary(self):
        """Display final comparison of all models"""
        print("\n" + "="*70)
        print("FINAL MODEL COMPARISON")
        print("="*70)
        
        # Sort by F1 score
        sorted_results = sorted(self.results, key=lambda x: x['f1'], reverse=True)
        
        print(f"\n{'Rank':<5} {'Model':<25} {'Accuracy':<10} {'Precision':<11} {'Recall':<10} {'F1':<10} {'Time':<8}")
        print("-" * 90)
        
        for i, result in enumerate(sorted_results, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            print(f"{emoji} {i:<3} {result['model']:<25} {result['accuracy']*100:>6.2f}%   "
                  f"{result['precision']*100:>6.2f}%    {result['recall']*100:>6.2f}%   "
                  f"{result['f1']*100:>6.2f}%   {result['train_time']:>6.1f}s")
        
        print("\n" + "="*70)
        print(f"🏆 BEST MODEL: {self.best_model_name}")
        print(f"   F1 Score: {self.best_score*100:.2f}%")
        print("="*70)
        
        # Save summary
        summary = {
            'best_model': self.best_model_name.lower().replace(' ', '_').replace('-', '_'),
            'best_f1': float(self.best_score),
            'all_results': sorted_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open('models/model_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print("\n✓ Summary saved to models/model_summary.json")
        print("✓ Best model artifacts saved in models/ directory")
        print("\nYou can now run the Flask app and it will automatically use the best model!")

def main():
    """Main training pipeline"""
    print("\n" + "🎯"*35)
    print("COMPREHENSIVE MODEL TRAINING SYSTEM")
    print("🎯"*35)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Load data (balanced for fair comparison)
    X_train, X_test, y_train, y_test = trainer.load_data(balance=True, max_samples=5000)
    
    # Train all ML models
    print("\n" + "🔬"*35)
    print("PHASE 1: CLASSICAL MACHINE LEARNING MODELS")
    print("🔬"*35)
    
    trainer.train_random_forest(X_train, X_test, y_train, y_test)
    trainer.train_xgboost(X_train, X_test, y_train, y_test)
    trainer.train_svm(X_train, X_test, y_train, y_test)
    trainer.train_logistic_regression(X_train, X_test, y_train, y_test)
    
    # Train neural networks
    if TORCH_AVAILABLE:
        print("\n" + "🧠"*35)
        print("PHASE 2: NEURAL NETWORK MODELS")
        print("🧠"*35)
        
        trainer.train_lstm(X_train, X_test, y_train, y_test)
    
    # Train transformers (optional - can be slow)
    if TRANSFORMERS_AVAILABLE:
        print("\n" + "🤖"*35)
        print("PHASE 3: TRANSFORMER MODELS (LLMs)")
        print("🤖"*35)
        
        # Ask user if they want to train transformers
        print("\n⚠️  Transformer training can take 10-30 minutes.")
        response = input("Do you want to train transformer models? (y/n): ").lower()
        
        if response == 'y':
            # Use smaller/faster models
            trainer.train_transformer(X_train, X_test, y_train, y_test, 'distilbert-base-uncased')
    
    # Display final summary
    trainer.display_final_summary()

if __name__ == '__main__':
    main()
