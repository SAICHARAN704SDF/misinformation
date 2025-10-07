"""
Quick SVM retraining with matching vectorizer
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
import joblib
from preprocessing import TextPreprocessor

print("Loading data...")
df_mis = pd.read_csv('data/misinfo_train.csv')
df_non = pd.read_csv('data/nonmisinfo_train.csv')

mis_texts = df_mis['text'].dropna().astype(str).tolist()
non_texts = df_non['text'].dropna().astype(str).tolist()

# Balance dataset
min_samples = min(len(mis_texts), len(non_texts))
non_texts_sampled = np.random.choice(non_texts, size=min_samples, replace=False).tolist()

texts = mis_texts + non_texts_sampled
labels = [1] * len(mis_texts) + [0] * len(non_texts_sampled)

print(f"Dataset: {len(mis_texts)} misinfo + {len(non_texts_sampled)} nonmisinfo")

# Preprocess
print("Preprocessing...")
preprocessor = TextPreprocessor()
texts_clean = [preprocessor.clean_text(t) for t in texts]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    texts_clean, labels, test_size=0.2, random_state=42, stratify=labels
)

# Vectorize with SAME settings as training pipeline
print("Creating TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"Feature dimensions: {X_train_vec.shape[1]}")

# Train SVM with balanced weights
print("Training Linear SVM...")
svm = LinearSVC(
    C=1.0,
    class_weight='balanced',
    dual=False,
    max_iter=2000,
    random_state=42
)
svm.fit(X_train_vec, y_train)

# Evaluate
y_pred = svm.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

print(f"\n{'='*60}")
print(f"SVM Performance:")
print(f"{'='*60}")
print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
print(f"Precision: {p:.4f} ({p*100:.2f}%)")
print(f"Recall:    {r:.4f} ({r*100:.2f}%)")
print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")

# Save with matching names
print("\nSaving model and vectorizer...")
joblib.dump(svm, 'models/svm.pkl')
joblib.dump(vectorizer, 'models/svm_tfidf.pkl')

print("✓ SVM model and vectorizer saved!")
print("✓ Feature dimensions match now")
print(f"✓ Model expects: {X_train_vec.shape[1]} features")
print(f"✓ Vectorizer produces: {X_train_vec.shape[1]} features")
