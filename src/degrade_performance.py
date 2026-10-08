import pandas as pd
import numpy as np
from .paths import PRODUCTION_DATA, INCOMING_DATA

def inject_drift(file_path=PRODUCTION_DATA,output_path=INCOMING_DATA,noise_level=1.5):
    df=pd.read_csv(file_path); cols=[c for c in df.columns if c!='target']
    rng=np.random.default_rng(42); df[cols]=df[cols]+rng.normal(0,noise_level,size=(len(df),len(cols)))
    df.to_csv(output_path,index=False); print('Successfully injected synthetic performance degradation into dataset.')

if __name__=='__main__': inject_drift()
