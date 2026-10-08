import shutil
from src.data_preprocessing import create_iris_data
from src.train_model import execute_training
from src.pipeline import promote_staging_model
from src.paths import PRODUCTION_DATA,STAGING_MODEL,INCOMING_DATA,PRODUCTION_DIR
def test_deploy_model():
    create_iris_data(); shutil.copyfile(PRODUCTION_DATA,INCOMING_DATA); execute_training(PRODUCTION_DATA,STAGING_MODEL); promote_staging_model(); assert (PRODUCTION_DIR/'model.joblib').exists(); assert (PRODUCTION_DIR/'model_manifest.json').exists()
