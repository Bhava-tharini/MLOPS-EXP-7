import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from .paths import DATA_DIR, PRODUCTION_DATA

def create_iris_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=['sepal_length','sepal_width','petal_length','petal_width'])
    df['target'] = iris.target
    df.to_csv(PRODUCTION_DATA, index=False)
    return df

def preprocess_data(file_path=PRODUCTION_DATA):
    df = pd.read_csv(file_path)
    X = df.drop(columns=['target']); y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    print(f'Data preprocessed successfully. Train shape: {X_train.shape}, Test shape: {X_test.shape}')
    return X_train,X_test,y_train,y_test

if __name__ == '__main__':
    create_iris_data(); preprocess_data()
