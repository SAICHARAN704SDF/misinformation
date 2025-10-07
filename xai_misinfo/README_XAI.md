# Multilingual Misinformation Detection with XAI

## Objectives
- Detect misinformation in multilingual and code-mixed text (tweets/posts).
- Support English + other languages without forced translation (use multilingual models).
- Provide model comparison across Classical ML, Deep Neural Networks, and Multilingual Transformers.
- Offer explainability via SHAP and LIME.
- Provide a desktop UI (Tkinter) for prediction, explanation, and feedback collection.

## Model Groups
1. Classical (>=3): SVM (LinearSVC), RandomForest, XGBoost
2. Deep Learning (>=3): LSTM, GRU, BiGRU (bidirectional variant)
3. Multilingual Transformers (>=2): any list via `--transformer_models` (e.g. `xlm-roberta-base,distilbert-base-multilingual-cased,bert-base-multilingual-cased`).

## Data Expected
A CSV with at least columns: `text`, `label` (0/1). You may keep separate files for train/test or rely on script to split.

## Directory Layout
```
xai_misinfo/
  data/
  models/
  explain/
  ui/
  train_pipeline.py
  evaluate_models.py
  inference.py
  ui/app_tk.py
  explain/shap_utils.py
  explain/lime_utils.py
```

## Training Pipeline
- Performs stratified split (default 90/10) but can also do K-fold (argument `--kfold` > 1).
- Trains each model type; stores metrics JSON + model artifacts.
- Selects best model (highest F1) and symlinks/copies to `models/best_model/`.

## Explainability
- SHAP: Works for tree (TreeExplainer), linear (KernelExplainer), and transformers (sampling) with fallback.
- LIME: TextExplainer used on raw text.

## UI Features
- Text entry -> Predict -> shows score, label, or "I don't know" if below threshold.
- Highlight important tokens using SHAP/LIME weights.
- Feedback buttons; incorrect label stored to `data/feedback.csv`.

## Quick Start
```
pip install -r requirements_xai.txt
# Train classical + RNN + single transformer
python train_pipeline.py --data data/dataset.csv --transformer_model xlm-roberta-base

# OR train multiple transformers
python train_pipeline.py --data data/dataset.csv --transformer_models xlm-roberta-base,distilbert-base-multilingual-cased,bert-base-multilingual-cased --transformer_epochs 1

python ui/app_tk.py
```

## Threshold Logic
Final probability >= 0.50 => Misleading
Probability 0.40–0.49 => "I don't know" (uncertainty band)
Probability < 0.40 => Not Misleading

## Future Enhancements
- Active learning loop
- Model drift monitoring
- Ensemble of best classical + transformer
