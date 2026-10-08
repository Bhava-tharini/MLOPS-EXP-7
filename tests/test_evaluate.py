import shutil
from src.data_preprocessing import create_iris_data
from src.train_model import execute_training
from src.pipeline import evaluate_model
from src.paths import PRODUCTION_DATA,CHAMPION_MODEL,INCOMING_DATA
def test_evaluate_model():
    create_iris_data(); execute_training(PRODUCTION_DATA,CHAMPION_MODEL); shutil.copyfile(PRODUCTION_DATA,INCOMING_DATA); m=evaluate_model(CHAMPION_MODEL,INCOMING_DATA); assert m['passed_quality_gate']
