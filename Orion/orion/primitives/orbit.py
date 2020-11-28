from orbit.models.dlt import DLTFull
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa import arima_model


class OrbitTAD(object):
    def __init__(self, value_column='value', time_column='timestamp', global_trend_option='linear'):
        print("ORBIT INIT:", global_trend_option)
        self.value_col = value_column
        self.time_col = time_column
        self.global_trend_option = global_trend_option
        self.model = DLTFull(response_col=self.value_col, date_col=self.time_col, seed=42, global_trend_option=global_trend_option)

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

    def fit(self, X):
        print('ORBIT FIT1:', X.shape, self.global_trend_option)
        X = self.monotonize(X)
        self.model.fit(df=X)

    def predict(self, X):
        print('ORBIT PREDICT:', X.shape, self.global_trend_option)
        X = self.monotonize(X)
        pred = self.model.predict(df=X)
        errors = np.abs(pred.prediction.values - X.value.values)
        print("ERRORS", np.mean(errors), np.max(errors), np.std(errors))
        x = list(X.timestamp)
        y1 = list(X.value)
        y2 = list(pred.prediction)
        plt.clf()
        plt.plot(x, y1)
        plt.plot(x, y2)
        plt.plot(x, errors)
        plt.savefig('orbit_predict.png')
        return errors, (list(X.timestamp.astype(int) // 10**9))       
        

