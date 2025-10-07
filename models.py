import os
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Optional heavy deps (transformers / torch / datasets)
_TRANSFORMERS_AVAILABLE = True
try:
    # Delay heavy imports until actually used; wrap references in functions
    from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
    from datasets import Dataset
    import torch
except Exception as _e:
    _TRANSFORMERS_AVAILABLE = False
    _TRANSFORMERS_IMPORT_ERROR = _e

class RandomForestModel:
    """Random Forest based misinformation detector"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.is_trained = False
        # Index in predict_proba output corresponding to misinformation class
        self.misinfo_class_index = 1  # default assumption
    
    def train(self, texts: list, labels: list, class_weight: str = 'balanced'):
        """Train the Random Forest model.

        Args:
            texts: List of input texts
            labels: Corresponding list of labels (0/1)
            class_weight: Strategy for handling imbalance (default 'balanced')
        """
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            # Vectorize text
            self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)
            
            # Train model
            self.model = RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
                class_weight=class_weight,
                max_depth=None,
                min_samples_leaf=1
            )
            self.model.fit(X_train_vec, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            self.is_trained = True
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
        except Exception as e:
            print(f"Error training Random Forest: {e}")
            return None
    
    def predict(self, text: str, threshold: float = 0.5):
        """Predict misinformation for a single text.

        Args:
            text: Input text
            threshold: Probability threshold for classifying as Misleading
        Returns:
            result label string, confidence (max prob), misinfo_prob
        """
        if not self.is_trained or self.model is None or self.vectorizer is None:
            return None, None, None
            
        try:
            vec = self.vectorizer.transform([text])
            prob = self.model.predict_proba(vec)[0]
            # Ensure orientation inferred (only once)
            if not hasattr(self, 'misinfo_class_index') or self.misinfo_class_index not in (0,1):
                self._infer_label_orientation_safely()
            misinfo_prob = float(prob[self.misinfo_class_index])
            confidence = float(max(prob))
            pred = 1 if misinfo_prob >= 0.5 else 0  # raw predicted label before threshold override
            # Apply threshold override
            pred_thresholded = 1 if misinfo_prob >= threshold else 0
            if pred != pred_thresholded:
                pred = pred_thresholded
            
            result = "misinformation" if pred == 1 else "nonmisinformation"
            return result, confidence, misinfo_prob
            
        except Exception as e:
            print(f"Error predicting with Random Forest: {e}")
            return None, None, None
    
    def save_model(self, model_path: str, vectorizer_path: str):
        """Save the trained model and vectorizer"""
        if self.is_trained:
            joblib.dump(self.model, model_path)
            joblib.dump(self.vectorizer, vectorizer_path)
    
    def load_model(self, model_path: str, vectorizer_path: str):
        """Load a pre-trained model and vectorizer"""
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            self.is_trained = True
            # Attempt to infer label orientation immediately after load
            self._infer_label_orientation_safely()
            return True
        except Exception as e:
            print(f"Error loading Random Forest model: {e}")
            return False

    def _infer_label_orientation_safely(self):
        """Try to detect which probability column represents misinformation.
        Heuristic: sample from data/misinfo_train.csv and data/nonmisinfo_train.csv if available.
        Compare mean probabilities for each class column. Assign the column whose mean is higher on misinfo samples
        (vs nonmisinfo) as the misinformation index. Falls back silently if anything fails.
        """
        try:
            if not self.is_trained or not hasattr(self.model, 'predict_proba'):
                return
            classes = getattr(self.model, 'classes_', None)
            if classes is None or len(classes) != 2:
                return
            # Quick exit if labels already conventional 0/1 and 1 likely misinfo
            # We'll still test but keep original if heuristic ambiguous.
            import pandas as _pd, numpy as _np
            mis_path = os.path.join('data','misinfo_train.csv')
            non_path = os.path.join('data','nonmisinfo_train.csv')
            if not (os.path.isfile(mis_path) and os.path.isfile(non_path)):
                # fallback: pick index where class value == 1 else last
                if 1 in classes:
                    self.misinfo_class_index = list(classes).index(1)
                else:
                    self.misinfo_class_index = 1
                return
            def _sample(path, n=80):
                try:
                    df = _pd.read_csv(path)
                    if 'text' not in df.columns:
                        return []
                    return df['text'].dropna().astype(str).head(n).tolist()
                except Exception:
                    return []
            mis_texts = _sample(mis_path)
            non_texts = _sample(non_path)
            if not mis_texts or not non_texts:
                if 1 in classes:
                    self.misinfo_class_index = list(classes).index(1)
                else:
                    self.misinfo_class_index = 1
                return
            # Vectorize
            X_mis = self.vectorizer.transform(mis_texts)
            X_non = self.vectorizer.transform(non_texts)
            probs_mis = self.model.predict_proba(X_mis)
            probs_non = self.model.predict_proba(X_non)
            # Compute mean per column
            mean_mis = probs_mis.mean(axis=0)
            mean_non = probs_non.mean(axis=0)
            # For each column, compute delta = mean_mis - mean_non; pick column with largest positive delta
            deltas = mean_mis - mean_non
            best_idx = int(_np.argmax(deltas))
            # FIX: Check absolute value; ensure we pick the column higher on misinfo
            # If best_idx has negative delta, it means column best_idx is LOWER on misinfo, so invert
            if deltas[best_idx] < 0:
                # The other column is the correct one
                best_idx = 1 - best_idx
            # Now ensure delta for chosen column is positive
            if deltas[best_idx] > 0.01:  # require meaningful difference
                self.misinfo_class_index = best_idx
            else:
                # fallback similar to before
                if 1 in classes:
                    self.misinfo_class_index = list(classes).index(1)
                else:
                    self.misinfo_class_index = 1
            print(f"[Orientation] Inferred misinformation probability column index: {self.misinfo_class_index} (deltas={deltas})")
        except Exception as _e:
            # Silent fail to avoid breaking prediction pipeline
            pass


class BERTModel:
    """BERT-based misinformation detector"""
    
    def __init__(self, model_name="bert-base-uncased"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.is_trained = False
        self.device = None
        if _TRANSFORMERS_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def prepare_data(self, texts: list, labels: list):
        """Prepare data for BERT training"""
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        train_df = pd.DataFrame({"text": train_texts, "label": train_labels})
        test_df = pd.DataFrame({"text": test_texts, "label": test_labels})
        
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)
        
        return train_dataset, test_dataset
    
    def tokenize_data(self, dataset):
        """Tokenize dataset for BERT"""
        def tokenize(batch):
            return self.tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)
        
        dataset = dataset.map(tokenize, batched=True)
        dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        return dataset
    
    def train(self, texts: list, labels: list, output_dir="./models/bert_model"):
        """Train the BERT model"""
        if not _TRANSFORMERS_AVAILABLE:
            print("[WARN] Transformers not available; skipping BERT training. Reason:", _TRANSFORMERS_IMPORT_ERROR)
            return None
        try:
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
            
            # Prepare datasets
            train_dataset, test_dataset = self.prepare_data(texts, labels)
            train_dataset = self.tokenize_data(train_dataset)
            test_dataset = self.tokenize_data(test_dataset)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=8,  # Reduced for memory
                per_device_eval_batch_size=8,
                num_train_epochs=2,  # Reduced for faster training
                weight_decay=0.01,
                logging_dir="./logs",
                logging_steps=50,
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
            )
            
            # Compute metrics function
            def compute_metrics(p):
                from sklearn.metrics import accuracy_score
                preds = p.predictions.argmax(-1)
                return {"accuracy": accuracy_score(p.label_ids, preds)}
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                tokenizer=self.tokenizer,
                compute_metrics=compute_metrics,
            )
            
            # Train
            trainer.train()
            results = trainer.evaluate()
            
            # Save model
            trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            self.is_trained = True
            return results
            
        except Exception as e:
            print(f"Error training BERT model: {e}")
            return None
    
    def predict(self, text: str):
        """Predict misinformation for a single text"""
        if not _TRANSFORMERS_AVAILABLE:
            return None, None
        if not self.is_trained or self.model is None or self.tokenizer is None:
            return None, None
            
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                pred = torch.argmax(probs).item()
                confidence = torch.max(probs).item()
            
            result = "misinformation" if pred == 1 else "nonmisinformation"
            return result, confidence
            
        except Exception as e:
            print(f"Error predicting with BERT: {e}")
            return None, None
    
    def load_model(self, model_path: str):
        """Load a pre-trained BERT model"""
        if not _TRANSFORMERS_AVAILABLE:
            print("[WARN] Transformers not available; cannot load BERT model. Reason:", _TRANSFORMERS_IMPORT_ERROR)
            return False
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.model = BertForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error loading BERT model: {e}")
            return False


class ModelEnsemble:
    """Ensemble of multiple models for better predictions"""
    
    def __init__(self):
        self.rf_model = RandomForestModel()
        self.bert_model = BERTModel()
        self.models_loaded = False
    
    def load_models(self, rf_model_path=None, rf_vectorizer_path=None, bert_model_path=None):
        """Load all available models"""
        rf_loaded = False
        bert_loaded = False
        
        if rf_model_path and rf_vectorizer_path:
            rf_loaded = self.rf_model.load_model(rf_model_path, rf_vectorizer_path)
        
        if bert_model_path:
            bert_loaded = self.bert_model.load_model(bert_model_path)
        
        self.models_loaded = rf_loaded or bert_loaded
        return rf_loaded, bert_loaded
    
    def predict(self, text: str):
        """Predict using ensemble of models"""
        if not self.models_loaded:
            return None, None, {}
        
        predictions = {}
        confidences = {}
        
        # Random Forest prediction
        if self.rf_model.is_trained:
            # Use default threshold here; caller can modify at higher layer if needed
            rf_pred, rf_conf, rf_misinfo_prob = self.rf_model.predict(text)
            if rf_pred is not None:
                predictions['RandomForest'] = rf_pred
                # Only include true model confidence in the confidence aggregation
                confidences['RandomForest'] = rf_conf
                # Store raw probability separately for threshold/UX purposes
                # (Previously we mixed this into confidences which forced averages toward ~0.5)
                rf_misinfo_prob_value = rf_misinfo_prob
        
        # BERT prediction
        if self.bert_model.is_trained:
            bert_pred, bert_conf = self.bert_model.predict(text)
            if bert_pred is not None:
                predictions['BERT'] = bert_pred
                confidences['BERT'] = bert_conf
        
        if not predictions:
            return None, None, {}
        
        # Ensemble prediction (majority vote with confidence weighting)
        misleading_votes = sum(1 for pred in predictions.values() if pred in ("misinformation","Misleading"))
        total_votes = len(predictions)

        # Weighted/aggregate confidence (simple mean for now) - excludes raw probability values
        avg_confidence = np.mean(list(confidences.values())) if confidences else 0.0

        final_prediction = "misinformation" if misleading_votes > total_votes / 2 else "nonmisinformation"

        details = {
            'individual_predictions': predictions,
            'individual_confidences': confidences,
            'misleading_votes': misleading_votes,
            'total_votes': total_votes
        }
        # Attach misinfo probability map if RF provided it
        if 'RandomForest' in predictions and 'rf_misinfo_prob_value' in locals():
            details['misinfo_probabilities'] = { 'RandomForest': rf_misinfo_prob_value }

        return final_prediction, avg_confidence, details


class GenericSklearnModel:
    """Wrapper for a generic sklearn-like classifier with an external TF-IDF vectorizer.
    Supports any estimator that implements predict(); if predict_proba not available but decision_function is,
    converts decision scores to probabilities via logistic transform for binary case.
    """
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.is_trained = False
        self.misinfo_class_index = 1  # default assumption
        self.threshold = 0.5  # default threshold

    def load(self, model_path: str, vectorizer_path: str, threshold_path: str = None) -> bool:
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            
            # Try to load threshold if path provided or auto-detect
            if threshold_path is None:
                # Auto-detect threshold file based on model path
                threshold_path = model_path.replace('.pkl', '_threshold.pkl')
            
            if threshold_path and os.path.exists(threshold_path):
                try:
                    self.threshold = joblib.load(threshold_path)
                    print(f"[GenericSklearnModel] Loaded threshold: {self.threshold:.3f}")
                except Exception as e:
                    print(f"[GenericSklearnModel] Could not load threshold ({e}), using default 0.5")
                    self.threshold = 0.5
            else:
                self.threshold = 0.5
            
            self.is_trained = True
            self._infer_label_orientation_safely()
            return True
        except Exception as e:
            print(f"[GenericSklearnModel] Failed loading {model_path}: {e}")
            return False

    def _binary_prob_from_decision(self, scores):
        import numpy as _np
        # Temperature scaling to reduce saturation
        temp = 2.5
        z = _np.clip(scores / temp, -15, 15)
        p = 1.0 / (1.0 + _np.exp(-z))
        # Clamp to avoid exact 0 / 1 which produced 100% confidence display
        p = _np.clip(p, 0.01, 0.99)
        return _np.vstack([1-p, p]).T  # shape (n,2)

    def predict(self, text: str, threshold: float = None):
        if not self.is_trained or self.model is None or self.vectorizer is None:
            return None, None, None
        
        # Use model's threshold if not explicitly provided
        if threshold is None:
            threshold = self.threshold if hasattr(self, 'threshold') else 0.5
            
        try:
            vec = self.vectorizer.transform([text])
            misinfo_prob = None
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(vec)[0]
                # Ensure orientation known
                if not hasattr(self, 'misinfo_class_index') or self.misinfo_class_index not in (0,1):
                    self._infer_label_orientation_safely()
                if len(proba) == 2:
                    misinfo_prob = float(proba[self.misinfo_class_index])
                else:
                    misinfo_prob = float(max(proba))
                confidence = float(max(proba))
            elif hasattr(self.model, 'decision_function'):
                scores = self.model.decision_function(vec)
                if scores.ndim == 1:
                    prob = self._binary_prob_from_decision(scores)[0]
                else:
                    # multi-class; softmax
                    import numpy as _np
                    ex = _np.exp(scores - scores.max())
                    prob = ex / ex.sum()
                if len(prob) == 2:
                    if not hasattr(self, 'misinfo_class_index') or self.misinfo_class_index not in (0,1):
                        self._infer_label_orientation_safely()
                    misinfo_prob = float(prob[self.misinfo_class_index])
                else:
                    misinfo_prob = float(max(prob))
                confidence = float(max(prob))
            else:
                # Fallback: use prediction only
                pred_label = self.model.predict(vec)[0]
                misinfo_prob = 1.0 if pred_label == 1 else 0.0
                confidence = 0.55  # neutral modest confidence when no probability info
            label_bin = 1 if misinfo_prob >= threshold else 0
            result = 'misinformation' if label_bin == 1 else 'nonmisinformation'
            return result, confidence, misinfo_prob
        except Exception as e:
            print(f"[GenericSklearnModel] predict error: {e}")
            return None, None, None

    def _infer_label_orientation_safely(self):
        """Heuristic orientation detection similar to RandomForestModel.
        Attempts to identify which probability column (0 or 1) corresponds to misinformation.
        """
        try:
            if not self.is_trained or not hasattr(self.model, 'predict_proba'):
                return
            classes = getattr(self.model, 'classes_', None)
            if classes is None or len(classes) != 2:
                return
            import pandas as _pd, numpy as _np
            mis_path = os.path.join('data','misinfo_train.csv')
            non_path = os.path.join('data','nonmisinfo_train.csv')
            if not (os.path.isfile(mis_path) and os.path.isfile(non_path)):
                if 1 in classes:
                    self.misinfo_class_index = list(classes).index(1)
                else:
                    self.misinfo_class_index = 1
                return
            def _sample(path, n=80):
                try:
                    df = _pd.read_csv(path)
                    if 'text' not in df.columns:
                        return []
                    return df['text'].dropna().astype(str).head(n).tolist()
                except Exception:
                    return []
            mis_texts = _sample(mis_path)
            non_texts = _sample(non_path)
            if not mis_texts or not non_texts:
                if 1 in classes:
                    self.misinfo_class_index = list(classes).index(1)
                else:
                    self.misinfo_class_index = 1
                return
            X_mis = self.vectorizer.transform(mis_texts)
            X_non = self.vectorizer.transform(non_texts)
            probs_mis = self.model.predict_proba(X_mis)
            probs_non = self.model.predict_proba(X_non)
            mean_mis = probs_mis.mean(axis=0)
            mean_non = probs_non.mean(axis=0)
            deltas = mean_mis - mean_non
            best_idx = int(_np.argmax(deltas))
            # FIX: If chosen column has negative delta, invert
            if deltas[best_idx] < 0:
                best_idx = 1 - best_idx
            if deltas[best_idx] > 0.01:
                self.misinfo_class_index = best_idx
            else:
                if 1 in classes:
                    self.misinfo_class_index = list(classes).index(1)
                else:
                    self.misinfo_class_index = 1
            print(f"[Orientation] (Generic) inferred misinfo probability column index: {self.misinfo_class_index} (deltas={deltas})")
        except Exception:
            pass

# ---------------- Torch Model Wrappers (RNN / CNN / Transformer) ---------------- #
class TorchTextModelWrapper:
    """Inference wrapper for simple torch models saved by train_pipeline (RNN/CNN/Transformer).
    Only loaded if summary best_model refers to matching artifact.
    """
    def __init__(self):
        self.model = None
        self.model_type = None
        self.device = None
        self.vocab = None
        self.tokenizer = None
        self.transformer_base_name = None
        self.is_trained = False
        if _TRANSFORMERS_AVAILABLE:
            import torch as _torch
            self.device = _torch.device('cuda' if _torch.cuda.is_available() else 'cpu')

    def _build_rnn(self, vocab_size, hidden_dim=128, rnn_type='lstm', bidirectional=False):
        import torch.nn as nn
        rnn_cls = {'lstm': nn.LSTM, 'gru': nn.GRU}[rnn_type]
        class _RNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, 128)
                self.rnn = rnn_cls(128, hidden_dim, batch_first=True, bidirectional=bidirectional)
                self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), 2)
            def forward(self, x):
                emb = self.embedding(x)
                out, _ = self.rnn(emb)
                out = out[:, -1, :]
                return self.fc(out)
        return _RNN()

    def _build_cnn(self, vocab_size):
        import torch.nn as nn
        class _CNN(nn.Module):
            def __init__(self):
                super().__init__()
                embed_dim=128; kernels=(3,4,5); filters=64
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self.convs = nn.ModuleList([nn.Conv1d(embed_dim, filters, k) for k in kernels])
                self.dropout = nn.Dropout(0.3)
                self.fc = nn.Linear(filters*len(kernels), 2)
            def forward(self, x):
                emb = self.embedding(x).transpose(1,2)
                feats = [torch.relu(c(emb)).max(dim=2)[0] for c in self.convs]
                cat = torch.cat(feats, dim=1)
                cat = self.dropout(cat)
                return self.fc(cat)
        return _CNN()

    def _build_transformer(self, base_name):
        from transformers import AutoModel
        import torch.nn as nn
        class _TModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = AutoModel.from_pretrained(base_name)
                hidden = self.base.config.hidden_size
                self.classifier = nn.Linear(hidden, 2)
            def forward(self, input_ids, attention_mask):
                out = self.base(input_ids=input_ids, attention_mask=attention_mask)
                pooled = out.last_hidden_state[:,0]
                return self.classifier(pooled)
        return _TModel()

    def load(self, models_dir: str, best_name: str) -> bool:
        try:
            import torch
            path = None
            if best_name.startswith('rnn_'):
                # Determine bidirectional
                bidirectional = best_name.endswith('bigru') or best_name.endswith('bi')
                path = os.path.join(models_dir, f"{best_name}.pt")
            elif best_name == 'cnn':
                path = os.path.join(models_dir, 'cnn.pt')
            elif best_name.startswith('transformer_'):
                path = os.path.join(models_dir, f"{best_name}.pt")
            else:
                return False
            if not os.path.exists(path):
                return False
            blob = torch.load(path, map_location='cpu')
            if best_name.startswith('rnn_'):
                self.vocab = blob.get('vocab')
                vocab_size = len(self.vocab or {})
                rnn_type = 'gru' if 'gru' in best_name else 'lstm'
                bidirectional = 'bigru' in best_name or (best_name.endswith('_bi'))
                self.model = self._build_rnn(vocab_size, rnn_type=rnn_type, bidirectional=bidirectional)
                self.model.load_state_dict(blob['model_state'])
                self.model_type = 'rnn'
            elif best_name == 'cnn':
                self.vocab = blob.get('vocab')
                vocab_size = len(self.vocab or {})
                self.model = self._build_cnn(vocab_size)
                self.model.load_state_dict(blob['model_state'])
                self.model_type = 'cnn'
            else:  # transformer
                self.transformer_base_name = blob.get('model_name')
                self.model = self._build_transformer(self.transformer_base_name)
                self.model.load_state_dict(blob['model_state'])
                from transformers import AutoTokenizer
                slug = best_name.replace('transformer_','')
                tok_dir = os.path.join(models_dir, f'tokenizer_{slug}')
                if os.path.isdir(tok_dir):
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(tok_dir)
                    except Exception:
                        self.tokenizer = AutoTokenizer.from_pretrained(self.transformer_base_name)
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.transformer_base_name)
                self.model_type = 'transformer'
            self.model.eval()
            if self.device:
                self.model.to(self.device)
            self.is_trained = True
            return True
        except Exception as e:
            print('[TorchTextModelWrapper] load error:', e)
            return False

    def _encode_vocab(self, text: str, max_len=100):
        tokens = text.lower().split()[:max_len]
        ids = []
        for w in tokens:
            ids.append(self.vocab.get(w, self.vocab.get('<unk>',1)))
        if len(ids) < max_len:
            ids += [0]*(max_len-len(ids))
        import torch
        return torch.tensor([ids], dtype=torch.long)

    def predict(self, text: str):
        if not self.is_trained:
            return None, None, None
        import torch
        try:
            if self.model_type in ('rnn','cnn'):
                x = self._encode_vocab(text)
                if self.device:
                    x = x.to(self.device)
                with torch.no_grad():
                    logits = self.model(x)
                    probs = torch.softmax(logits, dim=-1)[0]
                misinfo_prob = float(probs[1].cpu().item())
                confidence = float(torch.max(probs).cpu().item())
            else:  # transformer
                enc = self.tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
                if self.device:
                    enc = {k:v.to(self.device) for k,v in enc.items()}
                with torch.no_grad():
                    logits = self.model(enc['input_ids'], enc['attention_mask'])
                    probs = torch.softmax(logits, dim=-1)[0]
                misinfo_prob = float(probs[1].cpu().item())
                confidence = float(torch.max(probs).cpu().item())
            label = 'misinformation' if misinfo_prob >= 0.5 else 'nonmisinformation'
            return label, confidence, misinfo_prob
        except Exception as e:
            print('[TorchTextModelWrapper] predict error:', e)
            return None, None, None