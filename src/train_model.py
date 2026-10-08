import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from .paths import PRODUCTION_DATA, CHAMPION_MODEL, MODELS_DIR

def execute_training(data_path=PRODUCTION_DATA, output_model_path=CHAMPION_MODEL):
    df=pd.read_csv(data_path); X=df.drop(columns=['target']); y=df['target']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    model=RandomForestClassifier(n_estimators=100,random_state=42); model.fit(X_train,y_train)
    acc=accuracy_score(y_test,model.predict(X_test)); MODELS_DIR.mkdir(parents=True,exist_ok=True)
    with open(output_model_path,'wb') as f: pickle.dump(model,f)
    print(f'Retrained Model Production Accuracy evaluated: {acc:.4f}')
    return acc

if __name__ == '__main__': execute_training()
