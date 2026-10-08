from datetime import datetime,timedelta
import os,sys
from airflow import DAG
from airflow.operators.python import PythonOperator,BranchPythonOperator
PROJECT_ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'../..'))
if PROJECT_ROOT not in sys.path: sys.path.insert(0,PROJECT_ROOT)
from src.pipeline import ensure_base_artifacts,retrain_and_evaluate,promote_staging_model
from src.monitor_drift import evaluate_production_performance,DRIFT_THRESHOLD

def check_production_drift():
    ensure_base_artifacts(); acc=evaluate_production_performance()
    return 'trigger_retraining_sequence' if acc<DRIFT_THRESHOLD else 'skip_pipeline_execution'
def run_retraining(): print(f'Staging model accuracy: {retrain_and_evaluate():.4f}')
def notify_success(): print('SUCCESS: Automated Model Retraining Pipeline successfully replaced the active model artifact.')
args={'owner':'mlops_core','depends_on_past':False,'start_date':datetime(2026,1,1),'retries':1,'retry_delay':timedelta(minutes=1)}
with DAG('automated_model_retraining_pipeline',default_args=args,description='Production architecture monitoring performance thresholds.',schedule='@daily',catchup=False,tags=['mlops','retraining','airflow']) as dag:
    performance_check=BranchPythonOperator(task_id='evaluate_production_drift',python_callable=check_production_drift)
    retrain=PythonOperator(task_id='trigger_retraining_sequence',python_callable=run_retraining)
    promote=PythonOperator(task_id='promote_staging_to_production',python_callable=promote_staging_model)
    skip=PythonOperator(task_id='skip_pipeline_execution',python_callable=lambda: print('System operation stable. Retraining phase skipped.'))
    notify=PythonOperator(task_id='dispatch_success_notification',python_callable=notify_success)
    performance_check >> [retrain,skip]; retrain >> promote >> notify
