import re
import glob
import os
import pandas as pd
import numpy as np
from typing import Union, List, Optional, Tuple

try:
    # Optional lightweight translation (can be swapped for deep translators)
    # We avoid hard dependency; user can install 'googletrans==4.0.0rc1' or similar.
    from googletrans import Translator  # type: ignore
    _TRANSLATOR_AVAILABLE = True
    _translator_instance = Translator()
except Exception:
    _TRANSLATOR_AVAILABLE = False
    _translator_instance = None

class TextPreprocessor:
    """Text preprocessing utilities for misinformation detection.

    Enhancements:
    - Supports loading data from CSV or Excel (.xlsx) files.
    - Automatically discovers files if only directory paths are provided.
    - Normalizes text and validates inputs.
    """
    
    def __init__(self, force_english: bool = True, min_language_confidence: float = 0.0):
        # force_english: if True, attempt to translate non-English text to English
        # min_language_confidence reserved for future (some translators provide confidence)
        self.force_english = force_english
        self.min_language_confidence = min_language_confidence
    
    def clean_text(self, text: str, translate: bool = True) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            text = str(text)
            
        # Lowercase
        text = text.lower()
        
        # Optional translation (best-effort)
        if translate and self.force_english and _TRANSLATOR_AVAILABLE:
            try:
                # Detect + translate only if language not English
                detected = _translator_instance.detect(text)  # type: ignore
                if hasattr(detected, 'lang') and detected.lang and detected.lang.lower() != 'en':
                    text = _translator_instance.translate(text, dest='en').text  # type: ignore
            except Exception:
                # Silently continue if translation fails
                pass

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        
        # Remove mentions
        text = re.sub(r"@\w+", "", text)
        
        # Remove hashtags but keep the word
        text = re.sub(r"#", "", text)
        
        # Remove extra punctuation and special characters
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        
        # Remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()
        
        return text
    
    def batch_clean(self, texts: List[str], translate: bool = True) -> List[str]:
        """Clean multiple texts at once"""
        return [self.clean_text(text, translate=translate) for text in texts]
    
    def _read_file(self, path: str) -> pd.DataFrame:
        """Read a single file (CSV or Excel). Expects a 'text' column or single unnamed column.

        If no header, will treat first column as text. If a 'label' column is present we keep it
        but still override labels later according to misinfo/nonmisinfo grouping.
        """
        ext = os.path.splitext(path)[1].lower()
        if ext == '.csv':
            try:
                # Try reading with header first; if 'text' not found, read without header
                df = pd.read_csv(path)
                if 'text' not in df.columns:
                    df = pd.read_csv(path, header=None, names=['text'])
            except Exception:
                df = pd.read_csv(path, header=None, names=['text'])
        elif ext in ('.xlsx', '.xls'):
            try:
                df = pd.read_excel(path)
                if 'text' not in df.columns:
                    # Assume first column is text
                    first_col = df.columns[0]
                    df = df.rename(columns={first_col: 'text'})
            except Exception as e:
                raise RuntimeError(f"Failed to read Excel file {path}: {e}")
        else:
            raise ValueError(f"Unsupported file type: {ext} for {path}")
        # Ensure text column exists
        if 'text' not in df.columns:
            raise ValueError(f"No 'text' column found in file {path}")
        return df[['text']].copy()

    def _discover_file(self, pattern_candidates: List[str]) -> Optional[str]:
        """Given possible glob patterns, return the first existing file path or None."""
        for pattern in pattern_candidates:
            matches = glob.glob(pattern)
            if matches:
                return matches[0]
        return None

    def load_and_preprocess_data(self, misinfo_path: str, nonmisinfo_path: str) -> pd.DataFrame:
        """Load and preprocess training data from provided file paths.

        The paths can be:
        - Explicit file paths (csv/xlsx)
        - A directory containing files named like misinfo_train.* and nonmisinfo_train.*
        - Base names without extension (we try .csv then .xlsx)
        """
        try:
            # If a directory is provided instead of a file, try to auto-discover
            if os.path.isdir(misinfo_path):
                base_dir = misinfo_path
                misinfo_file = self._discover_file([
                    os.path.join(base_dir, 'misinfo_train.csv'),
                    os.path.join(base_dir, 'misinfo_train.xlsx'),
                    os.path.join(base_dir, 'misinfo.xlsx'),
                    os.path.join(base_dir, 'misinfo.csv')
                ])
                nonmisinfo_file = self._discover_file([
                    os.path.join(base_dir, 'nonmisinfo_train.csv'),
                    os.path.join(base_dir, 'nonmisinfo_train.xlsx'),
                    os.path.join(base_dir, 'nonmisinfo.xlsx'),
                    os.path.join(base_dir, 'nonmisinfo.csv'),
                    os.path.join(base_dir, 'true.csv'),
                    os.path.join(base_dir, 'true.xlsx')
                ])
                if not misinfo_file or not nonmisinfo_file:
                    raise FileNotFoundError("Could not auto-discover misinfo/nonmisinfo files in directory")
            else:
                # Attempt extension inference if no extension
                def resolve_path(p: str) -> str:
                    if os.path.isfile(p):
                        return p
                    # try csv then xlsx
                    if os.path.isfile(p + '.csv'):
                        return p + '.csv'
                    if os.path.isfile(p + '.xlsx'):
                        return p + '.xlsx'
                    return p  # return original; will error later
                misinfo_file = resolve_path(misinfo_path)
                nonmisinfo_file = resolve_path(nonmisinfo_path)

            if not os.path.isfile(misinfo_file):
                raise FileNotFoundError(f"Misinfo file not found: {misinfo_file}")
            if not os.path.isfile(nonmisinfo_file):
                raise FileNotFoundError(f"Non-misinfo file not found: {nonmisinfo_file}")

            misinfo_df = self._read_file(misinfo_file)
            nonmisinfo_df = self._read_file(nonmisinfo_file)

            # Assign labels (1 = misleading, 0 = not misleading / true)
            misinfo_df['label'] = 1
            nonmisinfo_df['label'] = 0

            df = pd.concat([misinfo_df, nonmisinfo_df], ignore_index=True)
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            df['clean_text'] = df['text'].astype(str).apply(lambda t: self.clean_text(t, translate=True))

            # Basic stats
            print(f"Loaded {len(df)} samples (misinfo={sum(df.label==1)}, true={sum(df.label==0)})")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def validate_input(self, text: str) -> bool:
        """Validate input text"""
        if not text or not isinstance(text, str):
            return False
        if len(text.strip()) < 5:
            return False
        return True