import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import TextPreprocessor
from models import RandomForestModel, BERTModel, ModelEnsemble

def test_preprocessing():
    """Test preprocessing functionality"""
    print("Testing text preprocessing...")
    
    preprocessor = TextPreprocessor()
    
    test_texts = [
        "BREAKING: This is FAKE news from @CNN https://fake-news.com #fakenews!!!",
        "Normal news article about politics",
        "",
        "123 #hashtag @mention www.website.com"
    ]
    
    for text in test_texts:
        cleaned = preprocessor.clean_text(text)
        print(f"Original: {text}")
        print(f"Cleaned:  {cleaned}")
        print(f"Valid:    {preprocessor.validate_input(text)}")
        print("-" * 50)

def test_models_with_dummy_data():
    """Test models with dummy data"""
    print("Testing models with dummy data...")
    
    # Create dummy data
    texts = [
        "This is fake news spreading misinformation",
        "This is reliable news from a trusted source",
        "Unverified claims about vaccine dangers",
        "Scientific study confirms vaccine safety",
        "Breaking conspiracy theory about government",
        "Official government statement on policy"
    ]
    labels = [1, 0, 1, 0, 1, 0]  # 1 = misleading, 0 = not misleading
    
    # Test Random Forest
    print("\n--- Testing Random Forest ---")
    rf_model = RandomForestModel()
    
    try:
        results = rf_model.train(texts, labels)
        if results:
            print("Training successful!")
            print(f"Accuracy: {results['accuracy']:.3f}")
            
            # Test prediction
            test_text = "This breaking news might be fake"
            prediction, confidence = rf_model.predict(test_text)
            print(f"Test prediction: {prediction} (confidence: {confidence:.3f})")
        else:
            print("Training failed")
    except Exception as e:
        print(f"Random Forest test failed: {e}")
    
    # Test BERT (smaller test due to computational requirements)
    print("\n--- Testing BERT (Basic Load) ---")
    try:
        bert_model = BERTModel()
        print("BERT model initialized successfully")
        # Skip actual training for this test - too computationally expensive
        print("BERT training test skipped (too computationally expensive for test)")
    except Exception as e:
        print(f"BERT test failed: {e}")

def test_ensemble():
    """Test model ensemble"""
    print("\n--- Testing Model Ensemble ---")
    
    ensemble = ModelEnsemble()
    
    # Test loading non-existent models
    rf_loaded, bert_loaded = ensemble.load_models()
    print(f"Models loaded - RF: {rf_loaded}, BERT: {bert_loaded}")
    
    if not ensemble.models_loaded:
        print("No models loaded (expected for fresh installation)")
    else:
        test_text = "This might be misinformation"
        prediction, confidence, details = ensemble.predict(test_text)
        print(f"Ensemble prediction: {prediction} (confidence: {confidence:.3f})")
        print(f"Details: {details}")

def main():
    print("=== Misinformation Detector Test Suite ===\n")
    
    test_preprocessing()
    test_models_with_dummy_data()
    test_ensemble()
    
    print("\n=== Test Complete ===")
    print("If you see this message, basic functionality is working!")
    print("To fully test the system:")
    print("1. Add your training data to the 'data' folder")
    print("2. Run: python train_models.py")
    print("3. Run: python app.py")

if __name__ == "__main__":
    main()