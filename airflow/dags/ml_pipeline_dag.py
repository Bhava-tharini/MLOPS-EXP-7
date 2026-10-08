from datetime import datetime
import os,sys,shutil
from airflow import DAG
from airflow.operators.python import PythonOperator
PROJECT_ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'../..'))
if PROJECT_ROOT not in sys.path: sys.path.insert(0,PROJECT_ROOT)
from src.data_preprocessing import create_iris_data,preprocess_data
from src.train_model import execute_training
from src.pipeline import ensure_base_artifacts,evaluate_model,promote_staging_model
from src.paths import PRODUCTION_DATA,CHAMPION_MODEL,INCOMING_DATA,STAGING_MODEL

def prep(): create_iris_data(); preprocess_data()
def train(): ensure_base_artifacts(); execute_training(PRODUCTION_DATA,CHAMPION_MODEL)
def evaluate():
    ensure_base_artifacts(); shutil.copyfile(PRODUCTION_DATA,INCOMING_DATA); m=evaluate_model(CHAMPION_MODEL,INCOMING_DATA)
    if not m['passed_quality_gate']: raise ValueError('Quality Gate FAILED')
def deploy(): execute_training(PRODUCTION_DATA,STAGING_MODEL); promote_staging_model()
with DAG('ml_pipeline_dag',start_date=datetime(2026,1,1),schedule='@daily',catchup=False,tags=['mlops','training']) as dag:
    preprocess_data_task=PythonOperator(task_id='preprocess_data_task',python_callable=prep)
    train_model_task=PythonOperator(task_id='train_model_task',python_callable=train)
    evaluate_model_task=PythonOperator(task_id='evaluate_model_task',python_callable=evaluate)
    deploy_model_task=PythonOperator(task_id='deploy_model_task',python_callable=deploy)
    preprocess_data_task >> train_model_task >> evaluate_model_task >> deploy_model_task
