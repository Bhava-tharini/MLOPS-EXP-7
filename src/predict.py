import pickle
import pandas as pd
from .paths import CHAMPION_MODEL

def predict(sample=None):
    sample=sample or [[5.1,3.5,1.4,0.2]]
    with open(CHAMPION_MODEL,'rb') as f: model=pickle.load(f)
    X=pd.DataFrame(sample,columns=['sepal_length','sepal_width','petal_length','petal_width']); pred=model.predict(X); prob=model.predict_proba(X)
    result=[{'class_id':int(pred[0]),'probabilities':[round(float(p),4) for p in prob[0]]}]; print('Sample Input:',sample); print('Prediction Output:',result); return result

if __name__=='__main__': predict()
