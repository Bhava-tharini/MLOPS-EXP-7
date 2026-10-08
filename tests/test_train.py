from src.data_preprocessing import create_iris_data
from src.train_model import execute_training
from src.paths import PRODUCTION_DATA,MODELS_DIR
def test_train_model():
    create_iris_data(); out=MODELS_DIR/'test_model.pkl'; acc=execute_training(PRODUCTION_DATA,out); assert out.exists(); assert 0<=acc<=1; out.unlink(missing_ok=True)
