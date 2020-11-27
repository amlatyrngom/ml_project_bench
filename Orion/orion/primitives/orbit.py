from orbit.models.dlt import DLTFull
import pandas as pd
import numpy as np


from statsmodels.tsa import arima_model


class OrbitTAD(object):
    def __init__(self, value_column='value', time_column='timestamp'):
        self.value_col = value_column
        self.time_col = time_column

    def predict(self, X):
        # arima_results = list()
        # dimensions = len(X.shape)

        # if dimensions > 2:
        #     raise ValueError("Only 1D o 2D arrays are supported")

        # if dimensions == 1 or X.shape[1] == 1:
        #     X = np.expand_dims(X, axis=0)

        # num_sequences = len(X)
        # i = 0
        # for sequence in range(num_sequences):
        #     arima = arima_model.ARIMA(X[sequence], order=(10, 1, 2))
        #     arima_fit = arima.fit(disp=0)
        #     forec = arima_fit.forecast(1)[0]
        #     arima_results.append(forec)
        #     i += 1
        #     if (i == 4):
        #         break

        # arima_results = np.asarray(arima_results)
        # print("ARIMA RESULTS: ", arima_results.shape)

        # if dimensions == 1:
        #     arima_results = arima_results[0]
        # print("ARIMA RESULTS: ", arima_results.shape)
        # return arima_results
        results = []
        print("ORBIT DIMENSIONS", X.shape)
        step_size = 1
        self.model = DLTFull(response_col=self.value_col, date_col=self.time_col, seasonality=52, seed=42)
        for iter_idx, seq in enumerate(X):
            print("Iteration {} out of {}".format(iter_idx + 1, len(X)))
            seq = seq.flatten()
            print("SEQ: ", seq.shape)
            df_data = {
                self.time_col: [i for i in range(len(seq))],
                self.value_col: seq,
            }
            df = pd.DataFrame(df_data)
            self.model.fit(df=df)
            test_df_data = {
                'value' : [37 for i in range(step_size)], # does not matter
                'timestamp': [i + len(seq) for i in range(step_size)]
            }
            test_df = pd.DataFrame(test_df_data)
            pred = self.model.predict(df=test_df)

            print("PREDICTED: ", pred.shape, pred)
            results.append([pred['prediction'][0]])
        results = np.asarray(results)
        print("ORBIT RESULTS:", results.shape)
        return results
        

