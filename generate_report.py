"""Generate a comprehensive report (metrics + dataset stats) for the misinformation detector.

Usage:
  python generate_report.py --misinfo data/misinfo.csv --nonmisinfo data/nonmisinfo.csv --output report.md

Outputs a Markdown report summarizing:
  - Dataset sizes / class balance
  - Basic text length stats
  - Sample cleaned examples
  - (If a trained model exists) evaluation metrics on a holdout split
  - Language translation coverage heuristic
"""
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from models import RandomForestModel
from preprocessing import TextPreprocessor


def build_report(misinfo_path: str, nonmisinfo_path: str, output: str):
    pre = TextPreprocessor()
    df = pre.load_and_preprocess_data(misinfo_path, nonmisinfo_path)
    if df.empty:
        raise SystemExit("Failed to load data for report.")

    # Heuristic translation coverage: count non-ascii original lines
    df['non_ascii'] = df['text'].apply(lambda t: any(ord(c) > 127 for c in str(t)))
    non_ascii_rate = df['non_ascii'].mean()

    # Train/test split for metrics
    X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

    # Train RF (fast) for evaluation (does not overwrite existing prod model)
    rf = RandomForestModel()
    rf_stats = rf.train(X_train.tolist(), y_train.tolist())
    # Predict on test
    preds = []
    probs = []
    for text in X_test:
        label, conf, misprob = rf.predict(text)
        if label is None:
            label = 'Not Misleading'
            conf = 0.0
        preds.append(1 if label == 'Misleading' else 0)
        probs.append(misprob)

    report = classification_report(y_test, preds, target_names=['Not Misleading','Misleading'], digits=4, zero_division=0)
    cm = confusion_matrix(y_test, preds)

    # Build markdown
    lines = []
    lines.append('# Misinformation Detection Report')
    lines.append('')
    lines.append('## Dataset Overview')
    lines.append(f'Total samples: {len(df)}')
    lines.append(f"Misinfo (label=1): {sum(df.label==1)} | Non-misinfo (label=0): {sum(df.label==0)}")
    imbalance_ratio = (sum(df.label==0) / max(1,sum(df.label==1))) if sum(df.label==1)>0 else float('inf')
    lines.append(f'Class imbalance ratio (non/mis): {imbalance_ratio:.2f}')
    lines.append(f'Non-ASCII (heuristic foreign language) rate: {non_ascii_rate*100:.2f}%')
    lines.append('')
    lines.append('## Text Length Stats (cleaned)')
    lengths = df['clean_text'].apply(lambda t: len(t.split()))
    lines.append(f'Mean words: {lengths.mean():.2f} | Median: {lengths.median():.0f} | Min: {lengths.min()} | Max: {lengths.max()}')
    lines.append('')
    lines.append('## Sample Cleaned Examples')
    sample = df.head(5)
    for i, row in sample.iterrows():
        lines.append(f'- ({row.label}) {row.clean_text[:160]}')
    lines.append('')
    lines.append('## Random Forest Holdout Metrics')
    if rf_stats:
        lines.append(f"Train split metrics: Accuracy={rf_stats['accuracy']:.4f} Precision={rf_stats['precision']:.4f} Recall={rf_stats['recall']:.4f} F1={rf_stats['f1_score']:.4f}")
    lines.append('')
    lines.append('### Classification Report (Test)')
    lines.append('')
    lines.append('```')
    lines.append(report)
    lines.append('```')
    lines.append('')
    lines.append('### Confusion Matrix (Test)')
    lines.append('')
    lines.append(str(cm))
    lines.append('')
    lines.append('## Notes')
    lines.append('- Misinfo probability threshold tuning may improve recall for minority class.')
    lines.append('- Consider oversampling or focal loss (if switching to neural models) to address imbalance.')
    lines.append('- For multilingual robustness, ensure translation dependency installed (googletrans) or integrate a multilingual model (e.g., XLM-R).')

    with open(output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'Report written to {output}')


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--misinfo', required=True, help='Path to misinfo file or directory')
    ap.add_argument('--nonmisinfo', required=True, help='Path to nonmisinfo (true) file or directory')
    ap.add_argument('--output', default='report.md', help='Markdown report output path')
    return ap.parse_args()


if __name__ == '__main__':
    args = parse_args()
    build_report(args.misinfo, args.nonmisinfo, args.output)
