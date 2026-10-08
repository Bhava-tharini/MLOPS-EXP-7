import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from .paths import CHAMPION_MODEL, INCOMING_DATA
DRIFT_THRESHOLD=0.80

def evaluate_production_performance():
    with open(CHAMPION_MODEL,'rb') as f: model=pickle.load(f)
    df=pd.read_csv(INCOMING_DATA); X=df.drop(columns=['target']); y=df['target']
    acc=accuracy_score(y,model.predict(X)); print(f'Operational production accuracy calculated at: {acc:.4f}'); return acc

if __name__=='__main__':
    print('RETRAINING REQUIRED' if evaluate_production_performance()<DRIFT_THRESHOLD else 'SYSTEM STABLE')
