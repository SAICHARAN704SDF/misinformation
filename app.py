import os
import json
import csv
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
from preprocessing import TextPreprocessor
from models import ModelEnsemble, RandomForestModel, BERTModel, GenericSklearnModel, TorchTextModelWrapper
from db import init_db, insert_prediction, add_feedback, fetch_history, export_history_rows
from generate_report import build_report
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from model_manager import ModelManager
try:
    from models import _TRANSFORMERS_AVAILABLE  # type: ignore
except Exception:
    _TRANSFORMERS_AVAILABLE = False
import io
import base64
ENABLE_ANALYTICS = True
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception as e:
    print(f"[WARN] Analytics disabled (matplotlib import failed): {e}")
    ENABLE_ANALYTICS = False

app = Flask(__name__)
# Configuration from environment for deployment
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-insecure-key')
USE_ONLY_RF = os.getenv('USE_ONLY_RF', 'false').lower() == 'true'
APP_DEBUG = os.getenv('FLASK_DEBUG', 'true').lower() == 'true'
RF_THRESHOLD = float(os.getenv('RF_THRESHOLD', '0.5'))

# Initialize components
preprocessor = TextPreprocessor()
model_ensemble = ModelEnsemble()
best_generic_model = GenericSklearnModel()
best_torch_model = TorchTextModelWrapper()
model_manager = ModelManager()
BEST_MODEL_NAME = None

DB_INITIALIZED = False
def ensure_db():
    global DB_INITIALIZED
    if not DB_INITIALIZED:
        init_db()
        DB_INITIALIZED = True

# Try to load pre-trained models
def load_models(force_model_id=None):
    """Load models; prefer active model from model_config.json, fallback to best from summary.
    
    Args:
        force_model_id: Optional model ID to force load (for API switching)
    """
    global BEST_MODEL_NAME
    rf_loaded = bert_loaded = False
    try:
        # Scan available models
        model_manager.scan_available_models()
        
        # Get active model from config or force_model_id
        if force_model_id:
            active_model = model_manager.get_model_details(force_model_id)
        else:
            active_model = model_manager.get_active_model()
        
        if active_model:
            print(f"[load_models] Active model config: {active_model['name']}")
            BEST_MODEL_NAME = active_model['id']
            model_file = active_model['model_file']
            vec_file = active_model.get('vectorizer_file')
            
            if active_model['type'] in ('ml', 'ensemble'):
                model_path = os.path.join('models', model_file)
                vec_path = os.path.join('models', vec_file) if vec_file else None
                
                if vec_path and os.path.exists(model_path) and os.path.exists(vec_path):
                    if best_generic_model.load(model_path, vec_path):
                        print(f"[load_models] ✓ Loaded: {active_model['name']}")
                        return True, False
                else:
                    print(f"[load_models] ✗ Files missing for {active_model['name']}")
            elif active_model['type'] == 'neural_network':
                if best_torch_model.load('models', active_model['id']):
                    print(f"[load_models] ✓ Loaded: {active_model['name']}")
                    return True, False
        
        # Fallback to summary-based loading
        summary_path = os.path.join('models','model_summary.json')
        # Legacy auto-conversion: if no model_summary but legacy RF exists, create summary & symlink copies
        if not os.path.isfile(summary_path):
            legacy_rf = os.path.join('models','random_forest_model.pkl')
            legacy_vec = os.path.join('models','tfidf_vectorizer.pkl')
            if os.path.isfile(legacy_rf) and os.path.isfile(legacy_vec):
                try:
                    import shutil
                    # Create generic copies expected by GenericSklearnModel loader if missing
                    generic_rf = os.path.join('models','random_forest.pkl')
                    generic_vec = os.path.join('models','tfidf.pkl')
                    if not os.path.isfile(generic_rf):
                        shutil.copyfile(legacy_rf, generic_rf)
                    if not os.path.isfile(generic_vec):
                        shutil.copyfile(legacy_vec, generic_vec)
                    with open(summary_path,'w',encoding='utf-8') as f:
                        json.dump({'best_model':'random_forest','best_f1':0.0,'auto_generated':True,'notice':'Recreate summary by retraining for accurate metrics'}, f, indent=2)
                    print('[load_models] Auto-generated model_summary.json for legacy RandomForest.')
                except Exception as auto_e:
                    print('[load_models] Auto-generation failed:', auto_e)
        if os.path.isfile(summary_path):
            try:
                with open(summary_path,'r',encoding='utf-8') as f:
                    summary = json.load(f)
                # Check both 'best_model' and 'best_model_file' keys
                BEST_MODEL_NAME = summary.get('best_model_file') or summary.get('best_model')
                print(f"[load_models] Best model from summary: {summary.get('best_model')} (file: {BEST_MODEL_NAME})")
            except Exception as se:
                print('[load_models] Failed reading model_summary.json:', se)
        # Attempt to load classical best model via GenericSklearnModel
        chosen_loaded = False
        if BEST_MODEL_NAME:
            # Map model names to their actual file names
            model_file_map = {
                'logisticregression': ('logistic.pkl', 'logistic_tfidf.pkl'),
                'linearsvm': ('svm.pkl', 'svm_tfidf.pkl'),
                'linear svm (tuned)': ('svm.pkl', 'svm_tfidf.pkl'),
                'logistic regression (tuned)': ('logistic.pkl', 'logistic_tfidf.pkl'),
                'random forest (tuned)': ('random_forest.pkl', 'tfidf.pkl'),
                'xgboost (tuned)': ('xgboost.pkl', 'xgboost_tfidf.pkl'),
                'gradient boosting (tuned)': ('gradient_boosting.pkl', 'tfidf.pkl'),
                'random_forest': ('random_forest.pkl', 'tfidf.pkl'),
                'randomforest': ('random_forest.pkl', 'tfidf.pkl'),
                'svm': ('svm.pkl', 'svm_tfidf.pkl'),
                'xgboost': ('xgboost.pkl', 'xgboost_tfidf.pkl'),
                'ensemble': ('ensemble.pkl', 'tfidf.pkl'),
                'logistic_regression': ('logistic.pkl', 'logistic_tfidf.pkl'),
                'gradient_boosting': ('gradient_boosting.pkl', 'tfidf.pkl')
            }
            model_name_lower = BEST_MODEL_NAME.lower()
            if model_name_lower in model_file_map:
                model_file, vec_file = model_file_map[model_name_lower]
                model_path = os.path.join('models', model_file)
                vec_path = os.path.join('models', vec_file)
                if os.path.exists(vec_path) and os.path.exists(model_path):
                    if best_generic_model.load(model_path, vec_path):
                        chosen_loaded = True
                        print(f"[load_models] ✓ Loaded best sklearn model: {BEST_MODEL_NAME}")
                else:
                    print(f"[load_models] ✗ Model files not found: {model_path}, {vec_path}")
        elif BEST_MODEL_NAME and (BEST_MODEL_NAME.startswith('rnn_') or BEST_MODEL_NAME.startswith('transformer_') or BEST_MODEL_NAME == 'cnn'):
            if best_torch_model.load('models', BEST_MODEL_NAME):
                chosen_loaded = True
                print(f"[load_models] ✓ Loaded best torch model: {BEST_MODEL_NAME}")
        
        # Return success status based on what was loaded
        if chosen_loaded:
            return True, False  # Model loaded successfully, no BERT
        
        # Fallback to legacy RF / BERT only if no modern model loaded
        print("[load_models] No modern model loaded, trying legacy models...")
        rf_model_path = os.path.join('models', 'random_forest_model.pkl')
        rf_vectorizer_path = os.path.join('models', 'tfidf_vectorizer.pkl')
        bert_model_path = os.path.join('models', 'bert_model')
        rf_loaded, bert_loaded = model_ensemble.load_models(
            rf_model_path if os.path.exists(rf_model_path) else None,
            rf_vectorizer_path if os.path.exists(rf_vectorizer_path) else None,
            bert_model_path if os.path.exists(bert_model_path) else None
        )
        return rf_loaded, bert_loaded
    except Exception as e:
        print(f"Error loading models: {e}")
        return False, False

@app.route('/')
def index():
    """Main page"""
    ensure_db()
    rf_loaded, bert_loaded = load_models()
    models_available = best_generic_model.is_trained or best_torch_model.is_trained or rf_loaded or bert_loaded
    return render_template('index.html', 
                         rf_loaded=False, 
                         bert_loaded=False,
                         models_available=models_available)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict misinformation for given text"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        debug = bool(data.get('debug'))
        threshold_override = data.get('threshold')
        effective_threshold = RF_THRESHOLD
        if isinstance(threshold_override,(int,float)) and 0 <= threshold_override <= 1:
            effective_threshold = float(threshold_override)
        
        if not text:
            return jsonify({'error': 'Please provide text to analyze'}), 400
        
        # Validate input
        if not preprocessor.validate_input(text):
            return jsonify({'error': 'Text is too short or invalid'}), 400
        
        # Clean (and possibly translate) text
        clean_text = preprocessor.clean_text(text, translate=True)
        # Basic language flag (very naive: if original has non-ascii and cleaned is ascii only we assume translation happened)
        original_language = 'en'
        if any(ord(c) > 127 for c in text):
            # Heuristic; real detection should use langdetect but we avoid extra dep
            original_language = 'non-en'
        
        # Ensure models loaded (especially on first request after restart)
        if not (best_generic_model.is_trained or best_torch_model.is_trained or model_ensemble.rf_model.is_trained or model_ensemble.bert_model.is_trained):
            load_models()

        if best_generic_model.is_trained:
            pred_label, conf, misinfo_prob = best_generic_model.predict(clean_text, threshold=effective_threshold)
            prediction = pred_label
            confidence = conf
            details = {
                'individual_predictions': { (BEST_MODEL_NAME or 'best_model'): pred_label },
                'individual_confidences': { (BEST_MODEL_NAME or 'best_model'): conf },
                'misinfo_probabilities': { (BEST_MODEL_NAME or 'best_model'): misinfo_prob }
            }
            misinfo_prob_override = misinfo_prob
        elif best_torch_model.is_trained:
            pred_label, conf, misinfo_prob = best_torch_model.predict(clean_text)
            prediction = pred_label; confidence = conf
            details = {
                'individual_predictions': { (BEST_MODEL_NAME or 'best_model'): pred_label },
                'individual_confidences': { (BEST_MODEL_NAME or 'best_model'): conf },
                'misinfo_probabilities': { (BEST_MODEL_NAME or 'best_model'): misinfo_prob }
            }
            misinfo_prob_override = misinfo_prob
        else:
            # Get prediction from ensemble
            prediction, confidence, details = model_ensemble.predict(clean_text)
            misinfo_prob_override = None

        if prediction is None:
            return jsonify({'error': 'No models available for prediction'}), 500

        # Extract misinfo probability BEFORE constructing history_entry
        misinfo_prob = None
        # New structure: details['misinfo_probabilities'] = { 'RandomForest': <prob> }
        if 'misinfo_probabilities' in details and isinstance(details['misinfo_probabilities'], dict):
            # If best generic model used, key may be BEST_MODEL_NAME
            if best_generic_model.is_trained:
                misinfo_prob = details['misinfo_probabilities'].get(BEST_MODEL_NAME or 'best_model')
            else:
                misinfo_prob = details['misinfo_probabilities'].get('RandomForest')
        # Backward compatibility (older structure stored inside individual_confidences)
        if misinfo_prob is None and 'individual_confidences' in details and 'RandomForest_misinfo_prob' in details['individual_confidences']:
            misinfo_prob = details['individual_confidences']['RandomForest_misinfo_prob']

        ensure_db()
        history_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'original_text': text,
            'clean_text': clean_text,
            'prediction': prediction,
            'confidence': float(confidence),
            'details': details,
            'misinfo_probability': misinfo_prob,
            'threshold_used': RF_THRESHOLD,
            'original_language': original_language,
            'details_json': json.dumps(details)
        }
        insert_prediction(history_entry)

        # Prepare response

        # Optionally append current evaluation metrics (if any corrected labels exist)
        hist = fetch_history()
        y_pred = []; y_true=[]
        label_map = {'misinformation':1,'Misleading':1,'nonmisinformation':0,'Not Misleading':0}
        for h in hist:
            pv = label_map.get(h['prediction'])
            if pv is None: continue
            tv = None
            if h.get('feedback_list'):
                for fb in h['feedback_list']:
                    if fb.get('correct_label') in ('misinformation','nonmisinformation'):
                        tv = label_map[fb['correct_label']]
            if tv is None: continue
            y_pred.append(pv); y_true.append(tv)
        eval_metrics = None
        if y_true:
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
            acc = accuracy_score(y_true, y_pred)
            p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
            cm = confusion_matrix(y_true, y_pred, labels=[1,0]).tolist()
            eval_metrics = {
                'samples_used': len(y_true),
                'accuracy': acc,
                'precision': p,
                'recall': r,
                'f1': f1,
                'confusion_matrix': {'labels':['misinformation','nonmisinformation'],'matrix': cm}
            }

        response = {
            'prediction': prediction,
            'confidence': float(confidence),
            'misinfo_probability': misinfo_prob,
            'threshold_used': RF_THRESHOLD,
            'details': details,
            'clean_text': clean_text,
            'original_language': original_language,
            'timestamp': history_entry['timestamp'],
            'evaluation': eval_metrics
        }

        # Debug diagnostics (feature importance snapshot / raw probabilities / threshold) 
        if debug:
            debug_payload = {
                'effective_threshold': effective_threshold,
                'best_model_name': BEST_MODEL_NAME,
                'model_type': ('sklearn' if best_generic_model.is_trained else ('torch' if best_torch_model.is_trained else 'ensemble')),
            }
            # Raw probability for RF legacy model
            if best_generic_model.is_trained and hasattr(best_generic_model.model,'predict_proba'):
                try:
                    vec = best_generic_model.vectorizer.transform([clean_text])
                    raw_proba = best_generic_model.model.predict_proba(vec)[0].tolist()
                    debug_payload['raw_proba'] = raw_proba
                except Exception as dp_e:
                    debug_payload['raw_proba_error'] = str(dp_e)
            # Feature importance contribution (approx) for RandomForest
            if model_ensemble.rf_model.is_trained and prediction is not None:
                try:
                    rf = model_ensemble.rf_model.model
                    vec = model_ensemble.rf_model.vectorizer
                    import numpy as _np
                    X = vec.transform([clean_text])
                    # Get non-zero feature indices
                    nz = X.nonzero()[1]
                    feature_names = vec.get_feature_names_out()
                    importances = rf.feature_importances_
                    contrib = sorted([(feature_names[i], float(importances[i])) for i in nz], key=lambda x: x[1], reverse=True)[:20]
                    debug_payload['top_feature_importances'] = contrib
                except Exception as fi_e:
                    debug_payload['feature_debug_error'] = str(fi_e)
            response['debug'] = debug_payload
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    """Record user feedback; optionally accept corrected label (misinformation/nonmisinformation)."""
    try:
        ensure_db()
        data = request.get_json()
        feedback_type = data.get('feedback')
        timestamp = data.get('timestamp')
        correct_label = data.get('correct_label')
        user_agent = request.headers.get('User-Agent')
        ip_addr = request.remote_addr
        if not feedback_type or not timestamp:
            return jsonify({'error': 'feedback and timestamp required'}), 400
        allowed_labels = {'misinformation','nonmisinformation', None}
        if correct_label not in allowed_labels:
            return jsonify({'error': 'Invalid correct_label'}), 400
        ok = add_feedback(timestamp, feedback_type, user_agent, ip_addr, correct_label=correct_label)
        if not ok:
            return jsonify({'error': 'Prediction not found for timestamp'}), 404
        # After recording feedback, compute up-to-date evaluation metrics (if possible)
        hist = fetch_history()
        y_pred = []
        y_true = []
        label_map = {'misinformation':1,'Misleading':1,'nonmisinformation':0,'Not Misleading':0}
        for h in hist:
            pred_val = label_map.get(h['prediction'])
            if pred_val is None:
                continue
            true_val = None
            if h.get('feedback_list'):
                for fb in h['feedback_list']:
                    if fb.get('correct_label') in ('misinformation','nonmisinformation'):
                        true_val = label_map[fb['correct_label']]
            if true_val is None:
                continue
            y_pred.append(pred_val); y_true.append(true_val)
        metrics_payload = None
        if y_true:
            acc = accuracy_score(y_true, y_pred)
            p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
            cm = confusion_matrix(y_true, y_pred, labels=[1,0]).tolist()
            metrics_payload = {
                'samples_used': len(y_true),
                'accuracy': acc,
                'precision': p,
                'recall': r,
                'f1': f1,
                'confusion_matrix': {'labels':['misinformation','nonmisinformation'],'matrix':cm}
            }
        return jsonify({'message': 'Feedback recorded successfully', 'correct_label': correct_label, 'evaluation': metrics_payload})
    except Exception as e:
        return jsonify({'error': f'Failed to record feedback: {str(e)}'}), 500

@app.route('/history')
def history():
    """Get prediction history"""
    ensure_db()
    return jsonify(fetch_history())

@app.route('/analytics')
def analytics():
    """Generate analytics visualization"""
    try:
        if not ENABLE_ANALYTICS:
            # Fallback summary (no charts)
            hist = fetch_history()
            if not hist:
                return jsonify({'error': 'No prediction data available'}), 400
            total = len(hist)
            misinfo_set = {'misinformation','Misleading'}
            misleading = sum(1 for h in hist if h['prediction'] in misinfo_set)
            not_mis = total - misleading
            avg_conf = sum(h['confidence'] for h in hist)/total if total else 0.0
            return jsonify({'summary': {
                'total': total,
                'misleading': misleading,
                'not_misleading': not_mis,
                'avg_confidence': avg_conf
            }})
        hist = fetch_history()
        if not hist:
            return jsonify({'error': 'No prediction data available'}), 400
        
        # Create visualizations
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Misinformation Detection Analytics', fontsize=16)
        
        # Extract data
        predictions = [entry['prediction'] for entry in hist]
        confidences = [entry['confidence'] for entry in hist]
        
        # 1. Prediction distribution
        pred_counts = pd.Series(predictions).value_counts()
        ax1.pie(pred_counts.values, labels=pred_counts.index, autopct='%1.1f%%', 
                colors=['#ff9999', '#66b3ff'])
        ax1.set_title('Prediction Distribution')
        
        # 2. Confidence distribution
        ax2.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Confidence Score Distribution')
        
        # 3. Predictions over time
        timestamps = [datetime.fromisoformat(entry['timestamp']) for entry in hist]
        misinfo_set = {'misinformation','Misleading'}
        misleading_over_time = [1 if pred in misinfo_set else 0 for pred in predictions]

        df_time = pd.DataFrame({'timestamp': timestamps, 'misleading': misleading_over_time})
        df_time = df_time.set_index('timestamp').resample('H').mean()

        ax3.plot(df_time.index, df_time['misleading'], marker='o')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Misleading Rate')
        ax3.set_title('Misleading Predictions Over Time')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Model performance comparison (if multiple models)
        model_stats = {}
        for entry in hist:
            if 'individual_predictions' in entry['details']:
                for model, pred in entry['details']['individual_predictions'].items():
                    if model not in model_stats:
                        model_stats[model] = {'misleading': 0, 'total': 0}
                    model_stats[model]['total'] += 1
                    if pred in misinfo_set:
                        model_stats[model]['misleading'] += 1
        
        if model_stats:
            models = list(model_stats.keys())
            misleading_rates = [model_stats[model]['misleading'] / model_stats[model]['total'] 
                              for model in models]
            
            ax4.bar(models, misleading_rates, color=['orange', 'lightgreen'])
            ax4.set_ylabel('Misleading Detection Rate')
            ax4.set_title('Model Performance Comparison')
        else:
            ax4.text(0.5, 0.5, 'No model comparison data available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Model Performance')
        
        plt.tight_layout()
        
        # Convert to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_string = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        
        return jsonify({'image': img_string})
        
    except Exception as e:
        return jsonify({'error': f'Failed to generate analytics: {str(e)}'}), 500

@app.route('/export_history')
def export_history():
    """Export prediction history to CSV"""
    try:
        rows = export_history_rows()
        if not rows:
            return jsonify({'error': 'No data to export'}), 400
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['timestamp','original_text','clean_text','prediction','confidence'])
        for r in rows:
            writer.writerow(r)
        
        # Create response
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'misinformation_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
        
    except Exception as e:
        return jsonify({'error': f'Export failed: {str(e)}'}), 500

@app.route('/train', methods=['POST'])
def train_models():
    """Train models with uploaded data"""
    try:
        # This would handle file uploads and training
        # For now, return a placeholder response
        return jsonify({
            'message': 'Training endpoint ready. Please implement file upload handling.',
            'status': 'placeholder'
        })
        
    except Exception as e:
        return jsonify({'error': f'Training failed: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    ensure_db()
    rf_loaded, bert_loaded = load_models()
    return jsonify({
        'status': 'healthy',
        'models': {
            'random_forest': rf_loaded,
            'bert': bert_loaded,
            'transformers_available': _TRANSFORMERS_AVAILABLE
        },
    'history_count': len(fetch_history())
    })

@app.route('/report', methods=['POST'])
def report():
    """Generate a markdown report from current data files (non-blocking simplified)."""
    try:
        # Expect paths in request or use defaults
        data = request.get_json(silent=True) or {}
        misinfo_path = data.get('misinfo', os.path.join('data','misinfo_train.csv'))
        nonmis_path = data.get('nonmisinfo', os.path.join('data','nonmisinfo_train.csv'))
        output = data.get('output', 'report.md')
        mode = data.get('mode','quick')  # quick | full

        if mode == 'quick':
            # Fast summary without re-training
            # Collect dataset sizes (if files exist)
            import pandas as pd
            mis_count = nonmis_count = 0
            if os.path.isfile(misinfo_path):
                try:
                    mis_count = sum(1 for _ in open(misinfo_path, 'r', encoding='utf-8', errors='ignore')) - 1 if 'text' in open(misinfo_path, 'r', encoding='utf-8', errors='ignore').readline().lower() else mis_count
                except Exception:
                    pass
            if os.path.isfile(nonmis_path):
                try:
                    nonmis_count = sum(1 for _ in open(nonmis_path, 'r', encoding='utf-8', errors='ignore')) - 1 if 'text' in open(nonmis_path, 'r', encoding='utf-8', errors='ignore').readline().lower() else nonmis_count
                except Exception:
                    pass
            hist = fetch_history()
            total_preds = len(hist)
            misinfo_set = {'misinformation','Misleading'}
            misleading_preds = sum(1 for h in hist if h['prediction'] in misinfo_set)
            avg_conf = (sum(h['confidence'] for h in hist)/total_preds) if total_preds else 0.0
            report_md = [
                '# Quick Misinformation Report',
                '',
                'Mode: quick (no re-training performed)',
                f'Dataset files present: misinfo={os.path.isfile(misinfo_path)} nonmisinfo={os.path.isfile(nonmis_path)}',
                f'Predictions recorded: {total_preds}',
                f'misinformation predictions: {misleading_preds}',
                f'Misleading rate: {(misleading_preds/total_preds*100 if total_preds else 0):.2f}%',
                f'Average confidence: {(avg_conf*100):.2f}%',
                '',
                '## Next Steps',
                '- Use full report mode for model holdout metrics (mode="full").',
                '- Gather more labeled samples if misleading rate seems off.',
                '- Adjust RF_THRESHOLD for sensitivity.'
            ]
            content = '\n'.join(report_md)
            # Write file
            with open(output,'w',encoding='utf-8') as f:
                f.write(content)
            return jsonify({'message':'Quick report generated','output':output,'content':content})
        else:
            build_report(misinfo_path, nonmis_path, output)
            with open(output,'r',encoding='utf-8') as f:
                content = f.read()
            return jsonify({'message':'Full report generated','output':output,'content':content[:5000]})
    except Exception as e:
        return jsonify({'error': f'Report generation failed: {e}'}), 500

@app.route('/evaluation')
def evaluation():
    """Compute evaluation metrics from stored history (using corrected labels when provided)."""
    try:
        hist = fetch_history()
        if not hist:
            return jsonify({'error':'No predictions to evaluate'}), 400
        y_pred = []
        y_true = []
        label_map = {'misinformation':1,'Misleading':1,'nonmisinformation':0,'Not Misleading':0}
        for h in hist:
            pred = label_map.get(h['prediction'])
            if pred is None:
                continue
            # If any feedback has a correct_label, prefer the latest
            true_label = None
            if h.get('feedback_list'):
                for fb in h['feedback_list']:
                    if fb.get('correct_label') and fb['correct_label'] in ('misinformation','nonmisinformation'):
                        true_label = label_map[fb['correct_label']]
            # If no correction, skip (cannot assume ground truth) to avoid inflating metrics with self-predictions
            if true_label is None:
                continue
            y_pred.append(pred)
            y_true.append(true_label)
        if not y_true:
            return jsonify({'error':'No corrected labels available yet for evaluation'}), 400
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[1,0]).tolist()
        return jsonify({
            'samples_used': len(y_true),
            'accuracy': acc,
            'precision': p,
            'recall': r,
            'f1': f1,
            'confusion_matrix': {
                'labels': ['misinformation','nonmisinformation'],
                'matrix': cm  # [[TP FN],[FP TN]] with ordering [1,0]
            }
        })
    except Exception as e:
        return jsonify({'error': f'Evaluation failed: {e}'}), 500

def create_app():
    """Factory to create app (useful for gunicorn/wsgi)"""
    return app

# ============================================================================
# MODEL MANAGEMENT API ROUTES
# ============================================================================

@app.route('/api/models/available', methods=['GET'])
def list_available_models():
    """Get list of all available models"""
    try:
        models = model_manager.scan_available_models()
        active = model_manager.get_active_model()
        
        return jsonify({
            'models': models,
            'active_model': active,
            'total': len(models)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/active', methods=['GET'])
def get_active_model_api():
    """Get currently active model"""
    try:
        model_manager.scan_available_models()
        active = model_manager.get_active_model()
        
        if active:
            return jsonify({'model': active, 'status': 'success'})
        else:
            return jsonify({'error': 'No active model set'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/activate', methods=['POST'])
def activate_model():
    """Switch to a different model"""
    try:
        data = request.get_json()
        model_id = data.get('model_id')
        
        if not model_id:
            return jsonify({'error': 'model_id required'}), 400
        
        model_manager.scan_available_models()
        
        if model_manager.set_active_model(model_id):
            # Reload models with the new active model
            success, _ = load_models(force_model_id=model_id)
            
            if success or best_generic_model.is_trained or best_torch_model.is_trained:
                return jsonify({
                    'message': f'Successfully activated model: {model_id}',
                    'model': model_manager.get_active_model(),
                    'status': 'success'
                })
            else:
                return jsonify({
                    'error': 'Model config updated but failed to load',
                    'status': 'partial'
                }), 500
        else:
            return jsonify({'error': f'Model not found: {model_id}'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/details/<model_id>', methods=['GET'])
def get_model_details_api(model_id):
    """Get detailed info about a specific model"""
    try:
        model_manager.scan_available_models()
        model = model_manager.get_model_details(model_id)
        
        if model:
            return jsonify({'model': model})
        else:
            return jsonify({'error': 'Model not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/models/manage')
def model_management_page():
    """Render model management page"""
    ensure_db()
    return render_template('model_manager.html')

# ============================================================================

if __name__ == '__main__':
    rf_loaded, bert_loaded = load_models()
    
    print("\n" + "="*70)
    if best_generic_model.is_trained:
        print(f"✓ BEST MODEL LOADED: {BEST_MODEL_NAME}")
        print(f"  Type: Advanced ML Classifier")
        print(f"  Status: Ready for predictions")
    elif best_torch_model.is_trained:
        print(f"✓ BEST MODEL LOADED: {BEST_MODEL_NAME}")
        print(f"  Type: Neural Network")
        print(f"  Status: Ready for predictions")
    elif rf_loaded or bert_loaded:
        print(f"⚠ Using Legacy Models:")
        print(f"  Random Forest: {'✓ Loaded' if rf_loaded else '✗ Not loaded'}")
        print(f"  BERT: {'✓ Loaded' if bert_loaded else '✗ Not loaded'}")
    else:
        print("✗ WARNING: No pre-trained models found!")
        print("  Please run: python xai_misinfo/train_pipeline.py")
    print("="*70 + "\n")
    
    app.run(debug=APP_DEBUG, host='0.0.0.0', port=int(os.getenv('PORT', '5000')))