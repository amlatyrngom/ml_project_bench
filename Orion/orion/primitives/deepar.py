from orbit.models.dlt import DLTFull
import pandas as pd
import numpy as np
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
import matplotlib.pyplot as plt


class DeepAR(object):
	def __init__(self,freq = "1H", h = 1 ,epochs =100, no_samples= 100, value_column= 'value', time_column = 'timestamp'):
		self.freq = freq
		self.h = h
		self.epochs = epochs
		self.no_samples = no_samples
		self.value_col = value_column
		self.time_col = time_column


	def fit(self, X):
		print('start fit')
		T,_ = X.shape  # number of time series
		X = X[[self.value_col]].values
		print(X.shape)
		prediction_length = self.h
		train_data = X.T
		h = self.h
		freq = self.freq
		self.start = pd.Timestamp("01-01-2019", freq=self.freq) 
		train_ds = ListDataset([{'target': x, 'start': self.start}
		                for x in train_data[:, :]],
		               freq=freq)
		
		estimator = DeepAREstimator(freq=self.freq, 
		                    prediction_length=self.h, context_length = 100,
		                    trainer=Trainer(epochs=self.epochs, learning_rate =  0.001))
		self.train_data = train_data
		self.predictor = estimator.train(train_ds)
		
	def predict(self, X):
		#initialise prediction array
		index = X[[self.time_col]].values
		X = X[[self.value_col]].values
		data_test = X.T
		predictions = np.zeros(data_test.shape)
		start_index = len(self.train_data)
		data = np.concatenate((self.train_data, data_test), axis = 1)
		# a hacjy solution to check if there is no time/train split
		if self.train_data.T.shape ==  X.shape:
			if np.sum((self.train_data.T - X)**2) < 1e-10:
				start_index = 0
				data = data_test
		
		windows = data.shape[1]-1
		list_ = []
		start = pd.Timestamp("01-01-2019", freq=self.freq) 
		for j in range(data.shape[0]):
		    list_ +=   [{"start": start, "target": np.array(data[j,:start_index+i*self.h])} for i in range(1,windows+1)]
		
		test_ds = ListDataset(list_,freq = self.freq)
		forecast_it, ts_it = make_evaluation_predictions(dataset=test_ds ,predictor=self.predictor,num_samples=self.no_samples)
		forecast_it = list(forecast_it)
		it = 0
		print('model predictions produced')
		for j in range(data.shape[0]):
		    for i in range(windows):
		        predictions[j,i*self.h:(i+1)*self.h] = forecast_it[it].mean
		        it+=1
		# error = np.abs((predictions - X.T))
		index 
		return predictions[:,:].T, X[:,:], index

