"""
Model Manager - Centralized model selection and switching
Allows easy switching between trained models via web UI or config
"""
import os
import json
from datetime import datetime
from typing import Dict, List, Optional

class ModelManager:
    """Manages all trained models and allows dynamic switching"""
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.config_path = os.path.join(models_dir, 'model_config.json')
        self.summary_path = os.path.join(models_dir, 'model_summary.json')
        self.active_model = None
        self.available_models = []
        
    def scan_available_models(self) -> List[Dict]:
        """Scan models directory and catalog all available models"""
        models = []
        
        if not os.path.exists(self.models_dir):
            return models
        
        # Define model patterns (model_file, vectorizer_file, display_name, model_type)
        model_patterns = [
            # ML Models
            ('svm.pkl', 'svm_tfidf.pkl', 'Linear SVM', 'ml'),
            ('logistic.pkl', 'logistic_tfidf.pkl', 'Logistic Regression', 'ml'),
            ('random_forest.pkl', 'tfidf.pkl', 'Random Forest', 'ml'),
            ('xgboost.pkl', 'xgboost_tfidf.pkl', 'XGBoost', 'ml'),
            ('gradient_boosting.pkl', 'tfidf.pkl', 'Gradient Boosting', 'ml'),
            ('ensemble.pkl', 'tfidf.pkl', 'Ensemble (Voting)', 'ml'),
            
            # Legacy models
            ('random_forest_model.pkl', 'tfidf_vectorizer.pkl', 'Random Forest (Legacy)', 'ml'),
        ]
        
        for model_file, vec_file, display_name, model_type in model_patterns:
            model_path = os.path.join(self.models_dir, model_file)
            vec_path = os.path.join(self.models_dir, vec_file)
            
            if os.path.exists(model_path) and os.path.exists(vec_path):
                # Get file size and modification time
                model_size = os.path.getsize(model_path) / 1024  # KB
                mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
                
                model_info = {
                    'id': model_file.replace('.pkl', ''),
                    'name': display_name,
                    'type': model_type,
                    'model_file': model_file,
                    'vectorizer_file': vec_file,
                    'size_kb': round(model_size, 2),
                    'modified': mod_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'available': True
                }
                
                # Try to get metrics from summary
                if os.path.exists(self.summary_path):
                    try:
                        with open(self.summary_path, 'r') as f:
                            summary = json.load(f)
                            # Find matching model in summary
                            for m in summary.get('all_models', []):
                                if display_name.lower() in m.get('name', '').lower():
                                    model_info['accuracy'] = m.get('accuracy', 0)
                                    model_info['f1'] = m.get('f1', 0)
                                    model_info['precision'] = m.get('precision', 0)
                                    model_info['recall'] = m.get('recall', 0)
                                    break
                    except:
                        pass
                
                models.append(model_info)
        
        # Scan for neural network models
        for filename in os.listdir(self.models_dir):
            if filename.endswith('.pt'):
                model_name = filename.replace('.pt', '')
                model_path = os.path.join(self.models_dir, filename)
                model_size = os.path.getsize(model_path) / 1024
                mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
                
                models.append({
                    'id': model_name,
                    'name': model_name.replace('_', ' ').title(),
                    'type': 'neural_network',
                    'model_file': filename,
                    'vectorizer_file': None,
                    'size_kb': round(model_size, 2),
                    'modified': mod_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'available': True
                })
        
        self.available_models = models
        return models
    
    def get_active_model(self) -> Optional[Dict]:
        """Get currently active model configuration"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    return config.get('active_model')
            except:
                pass
        
        # Default to best model from summary
        if os.path.exists(self.summary_path):
            try:
                with open(self.summary_path, 'r') as f:
                    summary = json.load(f)
                    best_model_file = summary.get('best_model_file', 'svm')
                    # Find in available models
                    for model in self.available_models:
                        if model['id'] == best_model_file:
                            return model
            except:
                pass
        
        return None
    
    def set_active_model(self, model_id: str) -> bool:
        """Set a model as active"""
        # Find model in available models
        model = None
        for m in self.available_models:
            if m['id'] == model_id:
                model = m
                break
        
        if not model:
            return False
        
        # Save to config
        config = {
            'active_model': model,
            'last_updated': datetime.now().isoformat(),
            'updated_by': 'user'
        }
        
        os.makedirs(self.models_dir, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.active_model = model
        return True
    
    def get_model_details(self, model_id: str) -> Optional[Dict]:
        """Get detailed information about a specific model"""
        for model in self.available_models:
            if model['id'] == model_id:
                return model
        return None
    
    def export_models_list(self) -> str:
        """Export list of all models as formatted string"""
        if not self.available_models:
            return "No models available"
        
        output = ["Available Models:", "=" * 80]
        active = self.get_active_model()
        
        for i, model in enumerate(self.available_models, 1):
            is_active = active and active['id'] == model['id']
            status = "★ ACTIVE" if is_active else ""
            
            output.append(f"\n{i}. {model['name']} {status}")
            output.append(f"   ID: {model['id']}")
            output.append(f"   Type: {model['type']}")
            output.append(f"   Size: {model['size_kb']} KB")
            output.append(f"   Modified: {model['modified']}")
            
            if 'accuracy' in model:
                output.append(f"   Metrics: Acc={model['accuracy']:.2%} | F1={model['f1']:.2%}")
        
        return "\n".join(output)


def main():
    """CLI interface for model management"""
    manager = ModelManager()
    print("Scanning for available models...")
    models = manager.scan_available_models()
    
    if not models:
        print("No models found! Please train models first.")
        return
    
    print(manager.export_models_list())
    
    active = manager.get_active_model()
    if active:
        print(f"\n✓ Current active model: {active['name']}")
    else:
        print("\n⚠ No active model set")
    
    print("\nTo change active model, use the web interface or call:")
    print("  manager.set_active_model('model_id')")


if __name__ == '__main__':
    main()
