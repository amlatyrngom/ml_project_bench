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

    def max_pool(self, df, l):
        if (df.timestamp.dtype != 'int'):
            t = list(df.timestamp.astype(int) // 10**9)
        else:
            t = df.timestamp
        v = df.value
        new_t = []
        new_v = []
        i = 0
        while i < len(v):
            start = i
            end = min(len(v), i + l)
            curr_t = t[start:end]
            curr_v = v[start:end]
            j = np.argmax(curr_v)
            if (i == 0): new_t.append(curr_t[0])
            else: new_t.append(curr_t[j])
            new_v.append(curr_v[i])
            i += l
        data = {
            'timestamp': pd.to_datetime(new_t, unit='s'),
            'value': new_v,
        }
        return pd.DataFrame(data)

    def process_input(self, df, l):
        return self.max_pool(self.monotonize(df), l)

    def fit(self, X):
        print('ORBIT FIT:  ', X.shape, self.global_trend_option)
        fig, axes = plt.subplots(nrows=2, ncols=1)
        axes[0].plot(X.timestamp, X.value)
        if (len(X) < 6000):
            self.l = 1
        elif (len(X) < 12000):
            self.l = 2
        else:
            self.l = 3
        X = self.process_input(X, self.l)
        axes[1].plot(X.timestamp, X.value)
        fig.tight_layout()
        plt.savefig('orbit_fit.png')

        print('ORBIT FIT AFTER PROCESS:  ', X.shape, self.global_trend_option)
        self.model.fit(df=X)

    def predict(self, X):
        print('ORBIT PREDICT:', X.shape, self.global_trend_option)
        X = self.process_input(X, self.l)
        pred = self.model.predict(df=X)
        errors = np.abs(pred.prediction.values - X.value.values)
        print("ERRORS", np.mean(errors), np.max(errors), np.std(errors))
        x = list(X.timestamp)
        y1 = list(X.value)
        y2 = list(pred.prediction)
        plt.clf()
        plt.plot(x, y1)
        plt.plot(x, y2)
        #plt.plot(x, errors)
        plt.savefig('orbit_predict.png')
        return errors, (list(X.timestamp.astype(int) // 10**9))       
        

