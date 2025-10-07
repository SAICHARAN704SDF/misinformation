#!/usr/bin/env python
"""Command-line interface for misinformation detection.

Usage examples (from project root, after training models):

  # Predict a single text
  .\.venv\Scripts\python.exe predict_cli.py --text "Breaking: vaccine turns people magnetic"

  # Predict multiple lines from a file
  .\.venv\Scripts\python.exe predict_cli.py --file samples.txt

  # Read from stdin (paste then Ctrl+Z+Enter on Windows)
  type some_text_file.txt | .\.venv\Scripts\python.exe predict_cli.py --stdin

  # JSON output
  .\.venv\Scripts\python.exe predict_cli.py --text "something" --json

  # Custom threshold
  .\.venv\Scripts\python.exe predict_cli.py --text "something" --threshold 0.35

If BERT model is present it will be used automatically (ensemble). Otherwise RandomForest only.
"""

import os
import sys
import json
import argparse
from datetime import datetime

from preprocessing import TextPreprocessor
from models import ModelEnsemble

DEFAULT_THRESHOLD = float(os.getenv("RF_THRESHOLD", "0.5"))


def load_models():
    ensemble = ModelEnsemble()
    rf_model_path = os.path.join('models', 'random_forest_model.pkl')
    rf_vec_path = os.path.join('models', 'tfidf_vectorizer.pkl')
    bert_model_path = os.path.join('models', 'bert_model')

    rf_loaded, bert_loaded = ensemble.load_models(
        rf_model_path if os.path.exists(rf_model_path) else None,
        rf_vec_path if os.path.exists(rf_vec_path) else None,
        bert_model_path if os.path.exists(bert_model_path) else None,
    )
    return ensemble, rf_loaded, bert_loaded


def predict_text(ensemble: ModelEnsemble, text: str):
    return ensemble.predict(text)


def parse_args():
    p = argparse.ArgumentParser(description='CLI misinformation detector')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--text', help='Single text to analyze')
    g.add_argument('--file', help='Path to a file with one text per line')
    g.add_argument('--stdin', action='store_true', help='Read lines from stdin')
    p.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD, help='Probability threshold for misinfo (applies to RandomForest probability if available)')
    p.add_argument('--json', action='store_true', help='Output JSON instead of plain text')
    p.add_argument('--show-prob', action='store_true', help='Show misinfo probability when available')
    return p.parse_args()


def apply_threshold_override(details: dict, label: str, threshold: float):
    # If we have RF misinfo probability, re-evaluate label
    if not details or 'individual_confidences' not in details:
        return label, None
    ic = details['individual_confidences']
    prob = ic.get('RandomForest_misinfo_prob')
    if prob is None:
        return label, None
    new_label = 'Misleading' if prob >= threshold else 'Not Misleading'
    return new_label, prob


def format_output(record, json_mode=False):
    if json_mode:
        return json.dumps(record, ensure_ascii=False)
    lines = [
        f"Text: {record['text']}",
        f"Prediction: {record['prediction']} (threshold={record['threshold']})",
        f"Confidence (avg): {record['confidence']:.4f}",
    ]
    if record.get('misinfo_probability') is not None:
        lines.append(f"RF misinfo probability: {record['misinfo_probability']:.4f}")
    if 'models' in record:
        for m, info in record['models'].items():
            lines.append(f"  - {m}: pred={info['prediction']} conf={info['confidence']:.4f}")
    return '\n'.join(lines)


def main():
    args = parse_args()
    pre = TextPreprocessor()
    ensemble, rf_loaded, bert_loaded = load_models()

    if not (rf_loaded or bert_loaded):
        print('ERROR: No models loaded. Train first: python train_models.py --data-path data --model rf', file=sys.stderr)
        sys.exit(1)

    inputs = []
    if args.text:
        inputs = [args.text]
    elif args.file:
        if not os.path.isfile(args.file):
            print(f"ERROR: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        with open(args.file, 'r', encoding='utf-8', errors='ignore') as f:
            inputs = [line.strip() for line in f if line.strip()]
    elif args.stdin:
        inputs = [line.strip() for line in sys.stdin if line.strip()]

    results = []

    for text in inputs:
        clean = pre.clean_text(text)
        pred, conf, details = predict_text(ensemble, clean)
        if pred is None:
            print('ERROR: Prediction failed (no models available).', file=sys.stderr)
            sys.exit(2)
        # Apply threshold override manually for CLI transparency
        final_label, prob = apply_threshold_override(details, pred, args.threshold)
        record = {
            'timestamp': datetime.utcnow().isoformat(),
            'text': text,
            'clean_text': clean,
            'prediction': final_label,
            'original_prediction': pred,
            'confidence': float(conf) if conf is not None else None,
            'misinfo_probability': prob,
            'threshold': args.threshold,
            'models': {},
        }
        if details and 'individual_predictions' in details:
            for mname, mpred in details['individual_predictions'].items():
                mconf = details['individual_confidences'].get(mname) if details.get('individual_confidences') else None
                record['models'][mname] = {
                    'prediction': mpred,
                    'confidence': float(mconf) if mconf is not None else None
                }
        results.append(record)

    if len(results) == 1:
        print(format_output(results[0], json_mode=args.json))
    else:
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            for r in results:
                print(format_output(r, json_mode=False))
                print('-' * 60)

if __name__ == '__main__':
    main()
