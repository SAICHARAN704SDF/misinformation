import os
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import TextPreprocessor
from models import RandomForestModel, BERTModel
import argparse

def train_random_forest(data_path):
    """Train Random Forest model"""
    print("Training Random Forest model...")
    
    # Initialize preprocessor and model
    preprocessor = TextPreprocessor()
    rf_model = RandomForestModel()
    
    # Load and preprocess data
    misinfo_file = os.path.join(data_path, "misinfo_train.csv")
    nonmisinfo_file = os.path.join(data_path, "nonmisinfo_train.csv")
    
    if not os.path.exists(misinfo_file) or not os.path.exists(nonmisinfo_file):
        print(f"Error: Data files not found in {data_path}")
        return False
    
    df = preprocessor.load_and_preprocess_data(misinfo_file, nonmisinfo_file)
    if df.empty:
        print("Error: No data loaded")
        return False
    
    print(f"Loaded {len(df)} samples")
    print(f"Misleading: {sum(df['label'] == 1)}, Not misleading: {sum(df['label'] == 0)}")
    
    # Train model
    texts = df['clean_text'].tolist()
    labels = df['label'].tolist()
    
    results = rf_model.train(texts, labels)
    if results:
        print("Random Forest Training Results:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}")
        
        # Save model
        os.makedirs("models", exist_ok=True)
        rf_model.save_model("models/random_forest_model.pkl", "models/tfidf_vectorizer.pkl")
        print("Random Forest model saved successfully!")
        return True
    else:
        print("Random Forest training failed!")
        return False

def train_bert(data_path):
    """Train BERT model"""
    print("Training BERT model...")
    
    # Initialize preprocessor and model
    preprocessor = TextPreprocessor()
    bert_model = BERTModel()
    
    # Load and preprocess data
    misinfo_file = os.path.join(data_path, "misinfo_train.csv")
    nonmisinfo_file = os.path.join(data_path, "nonmisinfo_train.csv")
    
    if not os.path.exists(misinfo_file) or not os.path.exists(nonmisinfo_file):
        print(f"Error: Data files not found in {data_path}")
        return False
    
    df = preprocessor.load_and_preprocess_data(misinfo_file, nonmisinfo_file)
    if df.empty:
        print("Error: No data loaded")
        return False
    
    print(f"Loaded {len(df)} samples")
    print(f"Misleading: {sum(df['label'] == 1)}, Not misleading: {sum(df['label'] == 0)}")
    
    # Train model (use original text for BERT, not cleaned)
    texts = df['text'].tolist()  # BERT can handle noisy text better
    labels = df['label'].tolist()
    
    # Take a smaller sample for faster training (remove this for full training)
    if len(texts) > 5000:
        print("Using sample of 5000 texts for faster training...")
        texts_sample, _, labels_sample, _ = train_test_split(
            texts, labels, train_size=5000, random_state=42, stratify=labels
        )
        texts, labels = texts_sample, labels_sample
    
    results = bert_model.train(texts, labels, output_dir="models/bert_model")
    if results:
        print("BERT Training Results:")
        for metric, value in results.items():
            print(f"  {metric}: {value}")
        print("BERT model saved successfully!")
        return True
    else:
        print("BERT training failed!")
        return False

def main():
    parser = argparse.ArgumentParser(description='Train misinformation detection models')
    parser.add_argument('--data-path', default='data', help='Path to training data directory')
    parser.add_argument('--model', choices=['rf', 'bert', 'both'], default='both', 
                       help='Which model to train (rf=Random Forest, bert=BERT, both=Both models)')
    
    args = parser.parse_args()
    
    print("=== Misinformation Detection Model Training ===")
    print(f"Data path: {args.data_path}")
    print("Supports CSV or Excel (.xlsx) files named misinfo_train / nonmisinfo_train.")
    print(f"Training: {args.model}")
    print()
    
    success = True
    
    if args.model in ['rf', 'both']:
        success &= train_random_forest(args.data_path)
        print()
    
    if args.model in ['bert', 'both']:
        success &= train_bert(args.data_path)
        print()
    
    if success:
        print("✅ Training completed successfully!")
        print("You can now run the web application with: python app.py")
    else:
        print("❌ Training failed!")
        print("Please check your data files and try again.")

if __name__ == "__main__":
    main()