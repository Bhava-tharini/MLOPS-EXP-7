from src.data_preprocessing import create_iris_data,preprocess_data
def test_preprocess_data():
    create_iris_data(); Xtr,Xte,ytr,yte=preprocess_data(); assert Xtr.shape==(120,4); assert Xte.shape==(30,4)
