"""
Test script to verify model predictions are working correctly
"""

import joblib
import numpy as np

print("=" * 70)
print("TESTING MODEL PREDICTIONS")
print("=" * 70)

# Load model and vectorizer
print("\nLoading model and vectorizer...")
model = joblib.load('models/logistic.pkl')
vectorizer = joblib.load('models/logistic_tfidf.pkl')
print("✓ Loaded successfully")

# Test cases - clear misinformation vs clear legitimate information
test_cases = [
    # Clear misinformation examples
    ("COVID-19 vaccines contain microchips to track people", "MISINFORMATION"),
    ("Bill Gates is using 5G towers to spread coronavirus", "MISINFORMATION"),
    ("Drinking bleach cures cancer and COVID-19", "MISINFORMATION"),
    ("The earth is flat and NASA is lying to everyone", "MISINFORMATION"),
    ("Vaccines cause autism in children", "MISINFORMATION"),
    
    # Clear legitimate information examples
    ("The World Health Organization recommends wearing masks during the pandemic", "NON-MISINFORMATION"),
    ("Studies show regular exercise improves cardiovascular health", "NON-MISINFORMATION"),
    ("Climate scientists report rising global temperatures", "NON-MISINFORMATION"),
    ("Researchers have developed new treatments for diabetes", "NON-MISINFORMATION"),
    ("The FDA approves medications after rigorous testing", "NON-MISINFORMATION"),
]

print("\n" + "=" * 70)
print("TESTING PREDICTIONS")
print("=" * 70)

predictions = []
probabilities = []

for i, (text, expected) in enumerate(test_cases, 1):
    # Vectorize
    X = vectorizer.transform([text])
    
    # Predict
    pred = model.predict(X)[0]
    pred_proba = model.predict_proba(X)[0]
    
    predictions.append(pred)
    probabilities.append(pred_proba)
    
    # Get probabilities
    prob_nonmisinfo = pred_proba[0] * 100
    prob_misinfo = pred_proba[1] * 100
    
    prediction_label = "MISINFORMATION" if pred == 1 else "NON-MISINFORMATION"
    match = "✓" if prediction_label == expected else "✗"
    
    print(f"\n{match} Test {i}: {expected}")
    print(f"   Text: {text[:60]}...")
    print(f"   Prediction: {prediction_label}")
    print(f"   Probabilities: Non-Misinfo={prob_nonmisinfo:.1f}%, Misinfo={prob_misinfo:.1f}%")

# Summary statistics
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

predictions_array = np.array(predictions)
unique, counts = np.unique(predictions_array, return_counts=True)

print(f"\nTotal predictions: {len(predictions)}")
print(f"Predictions breakdown:")
for val, count in zip(unique, counts):
    label = "MISINFORMATION" if val == 1 else "NON-MISINFORMATION"
    print(f"  {label}: {count} ({count/len(predictions)*100:.1f}%)")

# Check if model is predicting variety
if len(unique) == 1:
    print("\n⚠ WARNING: Model is predicting only ONE class!")
    print("   This indicates a serious problem with the model.")
elif len(unique) == 2:
    print(f"\n✓ Model is predicting both classes")
    print(f"  However, check if predictions match expectations above")

# Check probability distributions
all_probs = np.array(probabilities)
print(f"\nProbability statistics:")
print(f"  Non-Misinfo probabilities: min={all_probs[:,0].min():.3f}, max={all_probs[:,0].max():.3f}, mean={all_probs[:,0].mean():.3f}")
print(f"  Misinfo probabilities: min={all_probs[:,1].min():.3f}, max={all_probs[:,1].max():.3f}, mean={all_probs[:,1].mean():.3f}")

# Test with corrected samples
print("\n" + "=" * 70)
print("TESTING WITH CORRECTED SAMPLES")
print("=" * 70)

import pandas as pd

try:
    corrected_df = pd.read_csv('data/corrected_samples.csv')
    print(f"\nLoaded {len(corrected_df)} corrected samples")
    
    if len(corrected_df) > 0:
        X_corrected = vectorizer.transform(corrected_df['text'].fillna(''))
        y_true = corrected_df['corrected_label'].values
        y_pred = model.predict(X_corrected)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        
        print(f"\nMetrics on corrected samples:")
        print(f"  Accuracy: {accuracy*100:.2f}%")
        print(f"  Precision: {precision*100:.2f}%")
        print(f"  Recall: {recall*100:.2f}%")
        print(f"  F1 Score: {f1*100:.2f}%")
        print(f"\nConfusion Matrix:")
        print(f"  {cm}")
        print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
        print(f"  FN={cm[1,0]}, TP={cm[1,1]}")
        
        # Check predictions distribution
        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
        print(f"\nPredictions on corrected samples:")
        for val, count in zip(unique_pred, counts_pred):
            label = "MISINFORMATION" if val == 1 else "NON-MISINFORMATION"
            print(f"  {label}: {count} ({count/len(y_pred)*100:.1f}%)")
            
except FileNotFoundError:
    print("\n⚠ corrected_samples.csv not found")
except Exception as e:
    print(f"\n⚠ Error testing corrected samples: {e}")

print("\n" + "=" * 70)
