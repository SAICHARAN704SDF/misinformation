# 🔎 Misinformation Detector

An advanced AI-powered web application that detects misinformation using multiple machine learning models including BERT (deep learning) and Random Forest (traditional ML).

## Features

### 🤖 Multiple ML Models
- **BERT Model**: Deep learning approach using transformers
- **Random Forest**: Traditional ML with TF-IDF vectorization
- **Model Ensemble**: Combines predictions from multiple models

### 🌐 Web Interface
- Modern, responsive web interface
- Real-time text analysis
- Interactive prediction results
- Confidence scoring and visualization

### 📊 Analytics & Tracking
- Prediction history tracking
- Analytics dashboard with charts
- User feedback collection
- CSV export functionality
- Automatic (best-effort) translation of non-English text to English (optional)
- Misinfo probability threshold control (RF)
- Markdown report generation script

### 🔧 Advanced Features
- Text preprocessing and cleaning
- Model performance comparison
- Background training capabilities
- Health monitoring endpoints

## Installation

1. **Clone or create the project:**
   ```bash
   cd d:\misinformation
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your training data:**
   - Place your training data files in the `data/` folder:
     - `misinfo_train.csv` - Contains misleading/false information
     - `nonmisinfo_train.csv` - Contains reliable/true information
   - Each file should have one column with text data (no headers)

## Usage

### 1. Train Models

Train both models:
```bash
python train_models.py --data-path data --model both
```

Train only Random Forest:
```bash
python train_models.py --data-path data --model rf
```

Train only BERT:
```bash
python train_models.py --data-path data --model bert
```

### 2. Run the Web Application

```bash
python app.py
```

The application will be available at: `http://localhost:5000`

### 3. Test the System

Run basic functionality tests:
```bash
python test_system.py
```

### 4. Generate a Markdown Report

Create a dataset + metrics report (Random Forest quick evaluation):
```bash
python generate_report.py --misinfo data/misinfo_train.csv --nonmisinfo data/nonmisinfo_train.csv --output report.md
```
Output: `report.md` (class balance, holdout metrics, confusion matrix, examples).

### 5. Adjust Misinfo Probability Threshold

Set an environment variable before starting the app (example 0.35 to increase recall):
```bash
$env:RF_THRESHOLD="0.35"; python app.py
```
Higher = stricter (fewer positives), lower = more sensitive.

## File Structure

```
d:\misinformation/
├── app.py                 # Main Flask web application
├── models.py              # ML model implementations
├── preprocessing.py       # Text preprocessing utilities
├── train_models.py        # Model training script
├── test_system.py         # Test suite
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── data/                 # Training data directory
│   ├── misinfo_train.csv
│   └── nonmisinfo_train.csv
├── models/               # Trained models storage
│   ├── random_forest_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── bert_model/
├── templates/            # HTML templates
│   └── index.html
└── static/              # Static assets (CSS, JS, images)
```

## API Endpoints

### Prediction
- **POST /predict** - Analyze text for misinformation
  ```json
  {
    "text": "Your text to analyze"
  }
  ```

### Feedback
- **POST /feedback** - Submit user feedback
  ```json
  {
    "feedback": "helpful|not_helpful|unsure",
    "timestamp": "prediction_timestamp"
  }
  ```

### Analytics
- **GET /analytics** - Generate analytics visualizations
- **GET /history** - Get prediction history
- **GET /export_history** - Export history as CSV
- **GET /health** - System health check

## Model Details

### Random Forest Model
- **Features**: TF-IDF vectorization with 5000 features
- **Configuration**: 200 estimators, n-grams (1,2)
- **Preprocessing**: Text cleaning, URL/mention removal
- **Training Time**: ~1-5 minutes
- **Memory Usage**: Low

### BERT Model
- **Base Model**: bert-base-uncased
- **Max Length**: 128 tokens
- **Batch Size**: 8 (configurable)
- **Epochs**: 2 (configurable)
- **Training Time**: ~30-60 minutes (depending on data size)
- **Memory Usage**: High (GPU recommended)

## Data Format

Your training data should be in CSV format:

**misinfo_train.csv:**
```
"This is fake news about vaccines"
"Unverified claims about election fraud"
"Conspiracy theory text here"
```

**nonmisinfo_train.csv:**
```
"Verified news from reliable source"
"Scientific research findings"
"Official government statement"
```

## Configuration

### Model Training Parameters

**Random Forest** (in `models.py`):
```python
RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
```

**BERT** (in `models.py`):
```python
TrainingArguments(
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01
)
```

### Text Preprocessing

The preprocessing module handles:
- Lowercasing
- URL removal
- Social media mentions (@username)
- Hashtag cleaning
- Special character removal
- Whitespace normalization
- Optional translation (requires installing `googletrans==4.0.0rc1`). If unavailable, it silently skips translation.

### Translation
Non-English detection is heuristic (non-ASCII check) unless translator dependency is present. For higher quality multilingual support, consider integrating a multilingual model (e.g., `xlm-roberta-base`) or a more robust translation service.

### Feedback Persistence
User feedback is stored in-memory for the session. To persist across restarts, implement a small SQLite or CSV append in the `/feedback` endpoint. After submission the UI now refreshes history and shows an inline badge.
Additionally, feedback events are automatically appended to `feedback_log.csv` (path override with `FEEDBACK_CSV_PATH`). Each line includes prediction id, timestamp, feedback type, user agent, ip, and created_at.

## Performance Tips

1. **For BERT Training:**
   - Use GPU if available (CUDA)
   - Reduce batch size if running out of memory
   - Consider using smaller datasets for testing

2. **For Production:**
   - Use model caching
   - Implement request rate limiting
   - Consider model quantization for faster inference

3. **For Large Datasets:**
   - Use data streaming for training
   - Implement batch processing
   - Consider distributed training

## Troubleshooting

### Common Issues

1. **Memory errors during BERT training:**
   - Reduce batch size to 4 or 2
   - Use CPU instead of GPU if necessary
   - Train on smaller dataset first

2. **Model files not found:**
   - Ensure training completed successfully
   - Check `models/` directory exists
   - Run training script again

3. **Web interface not loading:**
   - Check if port 5000 is available
   - Verify all dependencies installed
   - Check console for error messages

### Error Logs

Check the console output for detailed error messages. The application provides comprehensive error handling and logging.

## Future Enhancements

- [ ] Support for multiple languages
- [ ] Real-time model updating
- [ ] Integration with news APIs
- [ ] Batch processing capabilities
- [ ] Advanced analytics dashboard
- [ ] Model comparison tools
- [ ] Automated model retraining

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Run the test suite: `python test_system.py`
3. Check console logs for detailed error messages
4. Create an issue with detailed error information

## Model Performance

The system achieves good performance on misinformation detection:
- **Random Forest**: Fast inference, good for production
- **BERT**: Higher accuracy, slower inference
- **Ensemble**: Best of both worlds

Performance metrics are displayed after training and can be monitored through the web interface.

## Deployment (Render.com)

If you only need Random Forest (faster, no large dependencies), use the provided `requirements_rf_only.txt`.

### Quick Render Setup
1. Push this repository to GitHub.
2. In Render, create a New Web Service and point it at the repo.
3. Use the included `render.yaml` for automatic configuration OR set manually:
    - Build Command:
       ```
       pip install -r requirements_rf_only.txt && python train_models.py --data-path data --model rf || echo "Skip training"
       ```
    - Start Command:
       ```
       gunicorn app:create_app --bind 0.0.0.0:$PORT --workers=2 --threads=4
       ```
4. Set environment variables:
    - `USE_ONLY_RF=true`
    - `FLASK_DEBUG=false`
    - `SECRET_KEY` (generate a secure string)

### Adding Models to Render
You can either:
1. Commit pre-trained `models/random_forest_model.pkl` & `models/tfidf_vectorizer.pkl` into the repo (private repos recommended) OR
2. Mount a persistent disk and train on first deploy (longer build time) OR
3. Provide a separate storage (S3, etc.) and download during build.

### Upgrading to BERT Later
1. Switch to `requirements.txt` in Render build command.
2. Ensure Python version pinned to a version with available wheels (e.g., 3.12.x).
3. Add GPU hosting or accept slower CPU inference.
4. Retrain and deploy both models (`--model both`).

### Environment Variables Summary
| Variable | Purpose | Example |
|----------|---------|---------|
| USE_ONLY_RF | Disable BERT path | true |
| FLASK_DEBUG | Disable debug in prod | false |
| SECRET_KEY | Session security | (random 32 chars) |
| PORT | Provided by Render | 10000+ |

### Health Check
Render health check path: `/health` returns JSON like:
```
{
   "status": "healthy",
   "models": {"random_forest": true, "bert": false, "transformers_available": false},
   "history_count": 0
}
```

### Scaling Advice
| Concern | Action |
|---------|--------|
| Cold start speed | Keep only RF model (fast load) |
| Memory usage | Avoid transformers until needed |
| Accuracy upgrade | Add DistilBERT first before full BERT |
| Persist history | Add lightweight SQLite or external DB |

## RandomForest-Only Mode

Set `USE_ONLY_RF=true` to ensure the app:
- Skips loading / reporting BERT
- Avoids transformer imports
- Reduces container size & build time

You can still later re-enable BERT by clearing that variable and switching requirements.