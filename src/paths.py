from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
PRODUCTION_DIR = PROJECT_ROOT / 'production_model'
PRODUCTION_DATA = DATA_DIR / 'production_data.csv'
INCOMING_DATA = DATA_DIR / 'incoming_batch.csv'
CHAMPION_MODEL = MODELS_DIR / 'champion_model.pkl'
STAGING_MODEL = MODELS_DIR / 'staging_model.pkl'
METRICS_FILE = MODELS_DIR / 'metrics.json'
MANIFEST_FILE = PRODUCTION_DIR / 'model_manifest.json'
