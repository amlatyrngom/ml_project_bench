from orbit.models.dlt import DLTFull
import pandas as pd
import numpy as np
from mssa.mssa import mSSA
import matplotlib.pyplot as plt


class mSSATAD(object):
	def __init__(self, rank=2, value_column= 'value', time_column = 'timestamp'):
		self.model = mSSA(rank = rank,  col_to_row_ratio = 1)
		self.value_col = value_column
		self.time_col = time_column


	def fit(self, X):
		self.model.update_model(X.loc[:,[self.value_col]])
		self.train = X
		self.last_timestamp = X.loc[:,self.time_col].values[-1]

	def predict(self, X):
		if self.last_timestamp ==  X.loc[:,self.time_col].values[-1]:
			print ('no split')
		else:
			self.model.update_model(X.loc[:,[self.value_col]])

		results = self.model.predict(self.value_col,X.index[0],X.index[-1])['Mean Predictions'].values
		error = np.abs(results- X[self.value_col].values)
		# plt.plot(X[self.value_col].values)
		# plt.plot(error)
		# plt.show()
		return error, X.index
