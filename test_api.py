#!/usr/bin/env python3
"""
Test the deployed misinformation detector with various examples
"""
import requests
import json

API_URL = "http://127.0.0.1:5000/predict"

test_cases = [
    # Known misinformation examples
    ("Russia launched a special military operation to denazify Ukraine", "misinformation", "Russian propaganda"),
    ("Ukraine bombed its own civilians in Bucha", "misinformation", "False flag narrative"),
    ("COVID vaccines contain microchips for tracking people", "misinformation", "Conspiracy theory"),
    ("Biden stole the 2020 election with fake ballots", "misinformation", "Election fraud lie"),
    ("5G towers cause coronavirus", "misinformation", "Tech conspiracy"),
    
    # Known non-misinformation (factual statements)
    ("The sky appears blue during daytime", "nonmisinformation", "Basic science fact"),
    ("Python is a programming language", "nonmisinformation", "Tech fact"),
    ("Water boils at 100 degrees Celsius at sea level", "nonmisinformation", "Physics fact"),
    ("Today is October 6, 2025", "nonmisinformation", "Current date"),
    ("The United Nations is an international organization", "nonmisinformation", "Political fact"),
]

print("="*80)
print("TESTING MISINFORMATION DETECTOR")
print("="*80)

correct = 0
total = len(test_cases)

for i, (text, expected, description) in enumerate(test_cases, 1):
    try:
        response = requests.post(API_URL, json={"text": text}, timeout=10)
        if response.status_code == 200:
            data = response.json()
            prediction = data['prediction']
            confidence = data['confidence']
            misinfo_prob = data.get('misinfo_probability', 0)
            
            is_correct = prediction == expected
            correct += is_correct
            
            status = "✓" if is_correct else "✗"
            
            print(f"\n[{i}/{total}] {status} {description}")
            print(f"  Text: '{text[:60]}...'")
            print(f"  Expected: {expected}")
            print(f"  Predicted: {prediction} (confidence: {confidence:.2%}, misinfo_prob: {misinfo_prob:.2%})")
            
            if not is_correct:
                print(f"  ⚠️  MISPREDICTION!")
        else:
            print(f"\n[{i}/{total}] ✗ ERROR: HTTP {response.status_code}")
            print(f"  {response.text}")
    except Exception as e:
        print(f"\n[{i}/{total}] ✗ ERROR: {e}")

print("\n" + "="*80)
print(f"RESULTS: {correct}/{total} correct ({correct/total*100:.1f}%)")
print("="*80)

if correct == total:
    print("🎉 PERFECT! All predictions correct!")
elif correct >= total * 0.8:
    print("✓ GOOD: Most predictions correct")
elif correct >= total * 0.6:
    print("⚠️  FAIR: Some predictions correct, needs improvement")
else:
    print("✗ POOR: Model needs retraining or threshold adjustment")
