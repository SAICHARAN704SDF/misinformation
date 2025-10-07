"""
Model Management API Routes
Add these routes to app.py for dynamic model switching
"""
from flask import jsonify, request
from model_manager import ModelManager
import os

# Initialize model manager
model_mgr = ModelManager()

def init_model_management_routes(app, reload_models_callback):
    """Initialize model management routes"""
    
    @app.route('/api/models/available', methods=['GET'])
    def list_available_models():
        """Get list of all available models"""
        try:
            models = model_mgr.scan_available_models()
            active = model_mgr.get_active_model()
            
            return jsonify({
                'models': models,
                'active_model': active,
                'total': len(models)
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/models/active', methods=['GET'])
    def get_active_model():
        """Get currently active model"""
        try:
            model_mgr.scan_available_models()
            active = model_mgr.get_active_model()
            
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
            
            model_mgr.scan_available_models()
            
            if model_mgr.set_active_model(model_id):
                # Trigger model reload
                success = reload_models_callback()
                
                if success:
                    return jsonify({
                        'message': f'Successfully activated model: {model_id}',
                        'model': model_mgr.get_active_model(),
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
    def get_model_details(model_id):
        """Get detailed info about a specific model"""
        try:
            model_mgr.scan_available_models()
            model = model_mgr.get_model_details(model_id)
            
            if model:
                return jsonify({'model': model})
            else:
                return jsonify({'error': 'Model not found'}), 404
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/models/manage')
    def model_management_page():
        """Render model management page"""
        return render_template('model_manager.html')
