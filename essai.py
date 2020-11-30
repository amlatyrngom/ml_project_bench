from time import time
from tqdm import tqdm 
import site
import sys
import pandas as pd 
import numpy as np
import ast 
import matplotlib.pyplot as plt

site.addsitedir('Orion/')
site.addsitedir('MLPrimitives/')
site.addsitedir('mSSA/')


from orion.benchmark import benchmark, _summarize_results_datasets
from Orion.orion.evaluation import CONTEXTUAL_METRICS as METRICS
from Orion.orion.evaluation import contextual_confusion_matrix
from functools import partial

if ('accuracy' in METRICS): del METRICS['accuracy']
METRICS['confusion_matrix'] = contextual_confusion_matrix
metrics = {k: partial(fun, weighted=False) for k, fun in METRICS.items()}

BUCKET = 'd3-ai-orion'
S3_URL = 'https://{}.s3.amazonaws.com/{}'

BENCHMARK_DATA = pd.read_csv(S3_URL.format(
    BUCKET, 'datasets.csv'), index_col=0, header=None).applymap(ast.literal_eval).to_dict()[1]

def make_hyperparams(datasets, rank):
    hyperparams = {}
    rank_dict = {"mssa": {"orion.primitives.mssa.mSSATAD#1": {"rank": rank}}}
    for k in datasets:
        hyperparams[k] = rank_dict
    return hyperparams


score_dataframes = []
summary_dataframes = []

ranks = [i for i in range(4, 5)]
datasets = {
    'custom': ['synthetic_5']
}

for rank in ranks:
    pipelines = ['mssa']
    data = datasets
    for k, v in data.items():
        print(len(v))
    print(data)
    hyperparameters = make_hyperparams(data, rank)
    scores = benchmark(pipelines=pipelines, datasets=data, metrics=metrics, rank='f1', hyperparameters=hyperparameters, detrend=True)
    scores['rank'] = rank
    score_dataframes.append(scores)
    scores['confusion_matrix'] = [str(x) for x in scores['confusion_matrix']]
    
    score_summary = _summarize_results_datasets(scores, metrics)
    score_summary['rank'] = rank
    summary_dataframes.append(score_summary)


pd.concat(score_dataframes, ignore_index=True).to_pickle("mssa_synthetic_scores.pkl")
pd.concat(summary_dataframes, ignore_index=True).to_pickle("mssa_synthetic_summaries.pkl")
