from orbit.models.dlt import DLTFull
import pandas as pd
import numpy as np


from statsmodels.tsa import arima_model


class OrbitTAD(object):
    def __init__(self, value_column='value', time_column='timestamp'):
        self.value_col = value_column
        self.time_col = time_column
        self.model = DLTFull(response_col=self.value_col, date_col=self.time_col, seasonality=52, seed=42)

    def monotonize(self, df):    
        df = df.copy()
        if (df.timestamp.dtype != 'int'):
            col = list(df.timestamp.astype(int) // 10**9)
        else:
            col = df.timestamp
        for i in range(1, len(col)):
            if col[i] <= col[i-1]:
                col[i] = col[i-1] + 1
        df.timestamp = pd.to_datetime(col, unit='s')
        return df

    def process_input(self, df):
        return self.monotonize(df)

    def predict(self, X):
        print(X.shape)

        #X = self.process_input(X)
        #self.model.fit(df=X)
        #pred = self.model.predict(df=X)

        #print("ORBIT RESULTS:", results.shape)
        return None
        

