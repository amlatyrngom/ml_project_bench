from fbprophet import Prophet
import pandas as pd
import numpy as np
from statsmodels.tsa import arima_model
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


class ProphetTAD(object):
    def __init__(self, value_column='value', time_column='timestamp'):
        self.value_col = value_column
        self.time_col = time_column
    
    def fit(self, X, index):

        print(X.shape, index.shape, index[:10])   
        print("X shape: ", X.shape)
        print("index shape: ", index.shape)
        dataf = pd.DataFrame()
        
        ### Add aribtirary timestamps  ###
        self.timestamps =  pd.date_range(start='1/1/2018', periods = len(X), freq = '1H')
        dataf['ds'] =  self.timestamps 
        ## uncomment if index is timestamp
        # dataf['ds'] = pd.to_datetime(index)
        print(dataf['ds'] )
        dataf['y'] = X[:,0]
        self.model = Prophet(changepoint_prior_scale=0.001 ,  yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True)
        total_length = len(X)
        
        ### Add aribtirary seasonality to Prophet ###
        for i in range(1,200,2):
            self.model.add_seasonality(name='weekly_%s'%i, period=total_length/i, fourier_order=5)
        # self.model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        # self.model.add_seasonality(name='daily', period=0.5, fourier_order=5)

        self.model.fit(dataf, verbose = True)


    def predict(self, X, index):
        #initialise prediction array
        predictions = np.zeros(X.shape)
        df = pd.DataFrame()
        ### Add same aribtirary timestamps  ###
        # 
        ### Fix when there is train/test split ####
        split = False
        if split:
            df['ds'] =  pd.date_range(start=self.timestamps[-1], periods = len(X), freq = '1H')
        else: df['ds'] = self.timestamps
        
        ## uncomment if index is timestamp
        # df['ds'] = pd.to_datetime(index)
        

        predictions[:,0] = self.model.predict(df)['yhat'].tolist()
        print(X.shape, predictions.shape)
        try: print(r2_score(X[:,:],predictions[:,:], ))
        except:pass
        return predictions, X[:,:]
        

