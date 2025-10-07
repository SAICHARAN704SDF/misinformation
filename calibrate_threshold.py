"""
Threshold Calibration Tool
Finds optimal decision threshold to minimize false positives while maintaining good recall.
"""
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, accuracy_score, f1_score, classification_report, confusion_matrix
import joblib

def load_data():
    """Load training data"""
    print("Loading data...")
    misinfo_path = os.path.join('data', 'misinfo_train.csv')
    nonmisinfo_path = os.path.join('data', 'nonmisinfo_train.csv')
    
    df_misinfo = pd.read_csv(misinfo_path)
    df_nonmisinfo = pd.read_csv(nonmisinfo_path)
    
    df_misinfo['label'] = 1
    df_nonmisinfo['label'] = 0
    
    df = pd.concat([df_misinfo, df_nonmisinfo], ignore_index=True)
    return df['text'].tolist(), df['label'].tolist()

def find_optimal_thresholds(model, vectorizer, texts, labels):
    """Find optimal thresholds using different strategies"""
    print("\n" + "="*70)
    print("THRESHOLD CALIBRATION")
    print("="*70)
    
    # Get predictions
    X = vectorizer.transform(texts)
    
    # Get probability scores
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)[:, 1]
    elif hasattr(model, 'decision_function'):
        scores = model.decision_function(X)
        # Convert to probabilities
        probs = 1 / (1 + np.exp(-scores))
    else:
        print("Model doesn't support probability predictions!")
        return
    
    # Calculate precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(labels, probs)
    
    # Strategy 1: Maximize F1 Score
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    best_f1_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5
    
    # Strategy 2: High Precision (minimize false positives)
    high_precision_idx = np.where(precisions >= 0.75)[0]
    if len(high_precision_idx) > 0:
        high_prec_threshold = thresholds[high_precision_idx[0]] if high_precision_idx[0] < len(thresholds) else 0.7
    else:
        high_prec_threshold = 0.7
    
    # Strategy 3: Balanced (equal precision and recall)
    diff = np.abs(precisions - recalls)
    balanced_idx = np.argmin(diff)
    balanced_threshold = thresholds[balanced_idx] if balanced_idx < len(thresholds) else 0.5
    
    # Strategy 4: High Recall (minimize false negatives)
    high_recall_idx = np.where(recalls >= 0.80)[0]
    if len(high_recall_idx) > 0:
        high_recall_threshold = thresholds[high_recall_idx[-1]] if high_recall_idx[-1] < len(thresholds) else 0.3
    else:
        high_recall_threshold = 0.3
    
    strategies = {
        'max_f1': {
            'threshold': float(best_f1_threshold),
            'description': 'Maximizes F1 Score (balance of precision & recall)',
            'precision': float(precisions[best_f1_idx]),
            'recall': float(recalls[best_f1_idx]),
            'f1': float(f1_scores[best_f1_idx])
        },
        'high_precision': {
            'threshold': float(high_prec_threshold),
            'description': 'High Precision (minimize false alarms - fewer wrong misinfo labels)',
            'target': 'Precision ≥ 75%'
        },
        'balanced': {
            'threshold': float(balanced_threshold),
            'description': 'Balanced (equal precision and recall)',
            'precision': float(precisions[balanced_idx]),
            'recall': float(recalls[balanced_idx])
        },
        'high_recall': {
            'threshold': float(high_recall_threshold),
            'description': 'High Recall (catch most misinformation, may have more false positives)',
            'target': 'Recall ≥ 80%'
        }
    }
    
    # Test each strategy
    print("\nTesting Different Threshold Strategies:\n")
    
    for strategy_name, strategy_info in strategies.items():
        threshold = strategy_info['threshold']
        preds = (probs >= threshold).astype(int)
        
        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, zero_division=0)
        cm = confusion_matrix(labels, preds)
        
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"Strategy: {strategy_name.upper()}")
        print(f"  Description: {strategy_info['description']}")
        print(f"  Threshold: {threshold:.4f}")
        print(f"  Accuracy:  {accuracy:.2%}")
        print(f"  Precision: {precision:.2%} (of predicted misinfo, {precision:.0%} are actually misinfo)")
        print(f"  Recall:    {recall:.2%} (catches {recall:.0%} of actual misinfo)")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  Confusion Matrix:")
        print(f"    True Negatives (correctly identified nonmisinfo): {tn}")
        print(f"    False Positives (nonmisinfo wrongly labeled misinfo): {fp} ⚠️")
        print(f"    False Negatives (misinfo wrongly labeled nonmisinfo): {fn}")
        print(f"    True Positives (correctly identified misinfo): {tp}")
        print()
    
    return strategies

def apply_threshold_to_model():
    """Apply the recommended threshold to model_summary.json"""
    summary_path = os.path.join('models', 'model_summary.json')
    
    if not os.path.exists(summary_path):
        print("model_summary.json not found!")
        return
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    # Load model
    model_name = summary.get('best_model', 'logisticregression')
    # Try different name variations
    possible_names = [model_name, model_name.replace('_', ''), 'logistic_regression', 'logistic']
    model_path = None
    for name in possible_names:
        path = os.path.join('models', f'{name}.pkl')
        if os.path.exists(path):
            model_path = path
            break
    
    # Find matching vectorizer
    vec_path = None
    if model_path:
        # Try model-specific vectorizer first
        base_name = os.path.basename(model_path).replace('.pkl', '')
        vec_candidates = [
            os.path.join('models', f'{base_name}_tfidf.pkl'),
            os.path.join('models', 'logistic_tfidf.pkl'),
            os.path.join('models', 'tfidf.pkl')
        ]
        for vec_candidate in vec_candidates:
            if os.path.exists(vec_candidate):
                vec_path = vec_candidate
                break
    
    if not model_path or not os.path.exists(model_path):
        print(f"Model file not found!")
        return
    
    if not vec_path or not os.path.exists(vec_path):
        print(f"Vectorizer file not found!")
        return
    
    print(f"Loading model: {model_path}")
    print(f"Loading vectorizer: {vec_path}")
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    
    # Load data and find thresholds
    texts, labels = load_data()
    strategies = find_optimal_thresholds(model, vectorizer, texts, labels)
    
    # Recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print("\nBased on your concern about false positives (nonmisinfo labeled as misinfo):")
    print("Use the 'high_precision' strategy with threshold:", strategies['high_precision']['threshold'])
    print("\nThis will:")
    print("  ✓ Reduce false alarms (fewer nonmisinfo incorrectly flagged)")
    print("  ✓ Increase confidence when labeling something as misinformation")
    print("  ⚠ May miss some actual misinformation (lower recall)")
    
    print("\n" + "="*70)
    choice = input("\nWhich strategy do you want to use? [max_f1/high_precision/balanced/high_recall] (default: high_precision): ").strip().lower()
    
    if not choice:
        choice = 'high_precision'
    
    if choice not in strategies:
        print(f"Invalid choice '{choice}'. Using 'high_precision'.")
        choice = 'high_precision'
    
    selected_threshold = strategies[choice]['threshold']
    
    # Update model_summary.json
    summary['threshold'] = selected_threshold
    summary['threshold_strategy'] = choice
    summary['threshold_info'] = strategies[choice]
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Updated model_summary.json with threshold: {selected_threshold:.4f}")
    print(f"   Strategy: {choice}")
    print("\nRestart the Flask app to apply the new threshold.")

if __name__ == '__main__':
    apply_threshold_to_model()
