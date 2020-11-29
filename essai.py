import site
import ast
import sys
from Orion.orion.evaluation import CONTEXTUAL_METRICS as METRICS
from Orion.orion.evaluation import contextual_confusion_matrix
from functools import partial
from Orion.orion.benchmark import _summarize_results_datasets
import os
import pandas as pd
from orion.benchmark import benchmark

site.addsitedir('Orion/')
site.addsitedir('MLPrimitives/')

S3_URL = 'https://{}.s3.amazonaws.com/{}'
BUCKET = 'd3-ai-orion'

BENCHMARK_PATH = os.path.join(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'),
    'benchmark'
)

BENCHMARK_DATA = pd.read_csv(S3_URL.format(
    BUCKET, 'datasets.csv'), index_col=0, header=None).applymap(ast.literal_eval).to_dict()[1]



pipelines = [
    'prophet'
]

# hyperparamters = {'MSL':{ 
# "orion.primitives.mssa.mSSATAD#1" :{'rank':50}}}

del METRICS['accuracy']
METRICS['confusion_matrix'] = contextual_confusion_matrix
metrics = {k: partial(fun, weighted=False) for k, fun in METRICS.items()}

data = {'realAWSCloudwatch': ['ec2_network_in_5abac7','ec2_cpu_utilization_ac20cd','ec2_disk_write_bytes_1ef3de']}
# print(list(BENCHMARK_DATA.keys()))

# data = {key: BENCHMARK_DATA[key][:]}
datasets_names = '_'.join(list(data.keys()))
scores = benchmark(pipelines=pipelines, datasets=data, metrics=metrics, rank='f1')
scores.to_csv(f'notebooks/results/scores_{pipelines[0]}_{datasets_names}_no_w.csv')
summary = _summarize_results_datasets(scores, metrics)
summary.to_csv(f'notebooks/results/summary_{pipelines[0]}_{datasets_names}_no_w.csv')
