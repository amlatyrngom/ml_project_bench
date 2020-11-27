import site
site.addsitedir('Orion/')
site.addsitedir('MLPrimitives/')
site.addsitedir('orbit/')
from orion.data import load_anomalies, load_signal
from orion.benchmark import _load_signal, _load_pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from orion.analysis import _load_pipeline, analyze
from orbit.models.dlt import DLTFull, LGTFull


signal = 'real_1'
train, test = _load_signal(signal, False)
truth = load_anomalies(signal)
#train.timestamp = pd.to_datetime(train.timestamp, unit='s')
#truth.start = pd.to_datetime(truth.start, unit='s')
#truth.end = pd.to_datetime(truth.end, unit='s')


train.plot('timestamp', 'value')
test.plot('timestamp', 'value')
print(train.head())

# from orbit.diagnostics.plot import plot_predicted_data
# test_times = [1420 + i + 1 for i in range(400)]
# test_data = {
#     'timestamp' : test_times
# }
# test_df = pd.DataFrame(test_data)
# test_df.timestamp = pd.to_datetime(test_df.timestamp, unit='D')

def monotonize(df):    
    df = df.copy()
    col = list(df.timestamp.astype(int) // 10**9)
    print('AFTER', col[0])
    for i in range(1, len(col)):
        if col[i] <= col[i-1]:
            col[i] = col[i-1] + 1
    df.timestamp = pd.to_datetime(col, unit='s')
    print('AFTER', col[0])
    return df

train = monotonize(train)
window_size = 500
step_size = 50

lo = 0
hi = window_size
model = DLTFull(response_col='value', date_col='timestamp')
num_iters = 10
results = []
train.timestamp = pd.to_datetime(train.timestamp, unit='s')
model.fit(df=train)
pred = model.predict(df=train, decompose=True)
X = list(train.timestamp)
Y1 = list(train.value)
Y2 = list(pred.prediction)
plt.clf()
plt.plot(X, Y1, label='expected')
plt.plot(X, Y2, label='predicted')
plt.legend(loc=2)
plt.savefig("orbit_predict.png")
