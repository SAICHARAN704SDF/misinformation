import os
import argparse
import pandas as pd
from collections import Counter
from preprocessing import TextPreprocessor

POSSIBLE_MISINFO_NAMES = [
    'misinfo_train.csv', 'misinfotrain_train.csv', 'misinfo.csv', 'misinformation.csv'
]
POSSIBLE_NONMISINFO_NAMES = [
    'nonmisinfo_train.csv', 'true.csv', 'nonmisinfo.csv', 'reliable.csv'
]

def find_file(base_dir: str, candidates):
    for name in candidates:
        path = os.path.join(base_dir, name)
        if os.path.isfile(path):
            return path
    return None

def load_raw_frames(data_path: str):
    misinfo_file = find_file(data_path, POSSIBLE_MISINFO_NAMES)
    nonmisinfo_file = find_file(data_path, POSSIBLE_NONMISINFO_NAMES)
    if misinfo_file is None:
        raise FileNotFoundError(f"Could not locate a misinfo file in {data_path}. Tried: {POSSIBLE_MISINFO_NAMES}")
    if nonmisinfo_file is None:
        raise FileNotFoundError(f"Could not locate a non-misinfo file in {data_path}. Tried: {POSSIBLE_NONMISINFO_NAMES}")

    # Attempt reading flexibly
    mis_df = pd.read_csv(misinfo_file, header=None, names=['text'])
    non_df = pd.read_csv(nonmisinfo_file, header=None, names=['text'])
    mis_df['label'] = 1
    non_df['label'] = 0
    df = pd.concat([mis_df, non_df], ignore_index=True)
    return df, misinfo_file, nonmisinfo_file

def top_tokens(texts, n=15):
    counter = Counter()
    for t in texts:
        for w in t.split():
            if len(w) > 2:
                counter[w] += 1
    return counter.most_common(n)

def main():
    parser = argparse.ArgumentParser(description='Analyze misinformation dataset')
    parser.add_argument('--data-path', default='data', help='Directory containing training CSV files')
    parser.add_argument('--samples', type=int, default=5, help='How many sample rows per class to show')
    parser.add_argument('--top', type=int, default=15, help='Top tokens per class')
    args = parser.parse_args()

    print('=== Dataset Analysis ===')
    print(f"Data path: {args.data_path}")

    pre = TextPreprocessor()

    df, mis_file, non_file = load_raw_frames(args.data_path)
    print(f"Loaded misinfo file: {mis_file}")
    print(f"Loaded non-misinfo file: {non_file}")
    print(f"Total samples: {len(df)}")

    # Clean
    df['clean_text'] = df['text'].astype(str).apply(pre.clean_text)
    df['length'] = df['clean_text'].str.split().apply(len)

    # Stats
    counts = df['label'].value_counts().to_dict()
    pos = counts.get(1, 0)
    neg = counts.get(0, 0)
    imbalance_ratio = f"1:{round(neg/pos, 2)}" if pos > 0 else 'undefined'

    print('\nClass Distribution:')
    print(f"  Misleading (1): {pos}")
    print(f"  True/Not (0):  {neg}")
    print(f"  Ratio (pos:neg) ≈ {imbalance_ratio}")

    print('\nLength (word count) Stats:')
    print(df.groupby('label')['length'].describe()[['mean','50%','min','max']])

    # Show samples
    print('\nSample Misleading Texts:')
    for i, row in df[df.label==1].head(args.samples).iterrows():
        print(f"  - {row.clean_text[:180]}")
    print('\nSample Non-Misleading Texts:')
    for i, row in df[df.label==0].head(args.samples).iterrows():
        print(f"  - {row.clean_text[:180]}")

    # Token frequencies
    pos_tokens = top_tokens(df[df.label==1]['clean_text'], n=args.top)
    neg_tokens = top_tokens(df[df.label==0]['clean_text'], n=args.top)

    print('\nTop Tokens (Misleading):')
    for tok, c in pos_tokens:
        print(f"  {tok:<18} {c}")

    print('\nTop Tokens (Non-Misleading):')
    for tok, c in neg_tokens:
        print(f"  {tok:<18} {c}")

    # Very basic imbalance recommendation
    if pos and neg/pos > 20:
        print('\n[Imbalance Warning] Your dataset is highly imbalanced.')
        print('Consider:')
        print('  - Down-sample majority class (0)')
        print('  - Up-sample minority class (1)')
        print('  - Use class_weight="balanced" (already applied in RF)')
        print('  - Adjust RF_THRESHOLD (e.g., 0.3)')

    print('\n=== Analysis Complete ===')
    print('Next: Train -> python train_models.py --data-path data --model rf')
    print('       Run  -> python app.py')

if __name__ == '__main__':
    main()
