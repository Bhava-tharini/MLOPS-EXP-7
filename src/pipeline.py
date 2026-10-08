import json, shutil, pickle
from datetime import datetime, timezone
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .paths import *
from .data_preprocessing import create_iris_data, preprocess_data
from .train_model import execute_training
QUALITY_THRESHOLD=0.85
DRIFT_THRESHOLD=0.80

def ensure_base_artifacts():
    MODELS_DIR.mkdir(parents=True,exist_ok=True); PRODUCTION_DIR.mkdir(parents=True,exist_ok=True)
    if not PRODUCTION_DATA.exists(): create_iris_data()
    if not CHAMPION_MODEL.exists(): execute_training(PRODUCTION_DATA,CHAMPION_MODEL)
    if not INCOMING_DATA.exists(): shutil.copyfile(PRODUCTION_DATA,INCOMING_DATA)

def evaluate_model(model_path=CHAMPION_MODEL,data_path=INCOMING_DATA):
    with open(model_path,'rb') as f: model=pickle.load(f)
    df=pd.read_csv(data_path); X=df.drop(columns=['target']); y=df['target']; pred=model.predict(X)
    m={'accuracy':round(float(accuracy_score(y,pred)),4),'precision':round(float(precision_score(y,pred,average='weighted',zero_division=0)),4),'recall':round(float(recall_score(y,pred,average='weighted',zero_division=0)),4),'f1_score':round(float(f1_score(y,pred,average='weighted',zero_division=0)),4),'accuracy_threshold':QUALITY_THRESHOLD}
    m['passed_quality_gate']=m['accuracy']>=QUALITY_THRESHOLD; MODELS_DIR.mkdir(exist_ok=True); METRICS_FILE.write_text(json.dumps(m,indent=2)); print(f'Model Evaluation Metrics: {m}'); return m

def retrain_and_evaluate():
    ensure_base_artifacts(); return execute_training(INCOMING_DATA,STAGING_MODEL)

def promote_staging_model():
    PRODUCTION_DIR.mkdir(exist_ok=True); shutil.copyfile(STAGING_MODEL,CHAMPION_MODEL); shutil.copyfile(STAGING_MODEL,PRODUCTION_DIR/'model.joblib')
    m=evaluate_model(STAGING_MODEL,INCOMING_DATA); shutil.copyfile(METRICS_FILE,PRODUCTION_DIR/'metrics.json')
    (PRODUCTION_DIR/'training_metadata.json').write_text(json.dumps({'model_type':'RandomForestClassifier','random_state':42},indent=2))
    MANIFEST_FILE.write_text(json.dumps({'status':'DEPLOYED','deployment_timestamp':datetime.now(timezone.utc).isoformat(),'version':'v1.0.0','deployed_files':['model.joblib','metrics.json','training_metadata.json'],'metrics':m},indent=2))
    print('Staging configuration successfully promoted to live production champion model.')

def run_local_pipeline():
    print('>>> STAGE 1: Preprocessing Data...'); create_iris_data(); preprocess_data()
    print('>>> STAGE 2: Training Model...'); execute_training(PRODUCTION_DATA,CHAMPION_MODEL)
    print('>>> STAGE 3: Preparing Incoming Batch...'); shutil.copyfile(PRODUCTION_DATA,INCOMING_DATA)
    print('>>> STAGE 4: Evaluating Model & Validating Quality Gate...'); m=evaluate_model(CHAMPION_MODEL,INCOMING_DATA)
    if not m['passed_quality_gate']: raise RuntimeError('Quality Gate FAILED')
    print('>>> STAGE 5: Deploying Validated Model...'); shutil.copyfile(CHAMPION_MODEL,PRODUCTION_DIR/'model.joblib'); shutil.copyfile(METRICS_FILE,PRODUCTION_DIR/'metrics.json'); (PRODUCTION_DIR/'training_metadata.json').write_text(json.dumps({'model_type':'RandomForestClassifier','random_state':42},indent=2)); MANIFEST_FILE.write_text(json.dumps({'status':'DEPLOYED','deployment_timestamp':datetime.now(timezone.utc).isoformat(),'version':'v1.0.0','deployed_files':['model.joblib','metrics.json','training_metadata.json'],'metrics':m},indent=2)); print('SUCCESS: All pipeline stages completed successfully!')
