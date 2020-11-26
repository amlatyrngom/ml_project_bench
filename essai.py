import site
import sys
from Orion.orion.evaluation import CONTEXTUAL_METRICS as METRICS
from Orion.orion.evaluation import contextual_confusion_matrix
from functools import partial
from Orion.orion.benchmark import _summarize_results_datasets

site.addsitedir('Orion/')
site.addsitedir('MLPrimitives/')

from orion.benchmark import benchmark

pipelines = [
    'mssa'
]

# hyperparamters = {'MSL':{ 
# "orion.primitives.mssa.mSSATAD#1" :{'rank':50}}}



del METRICS['accuracy']
METRICS['confusion_matrix'] = contextual_confusion_matrix
metrics = {k: partial(fun, weighted=False) for k, fun in METRICS.items()}



scores = benchmark(pipelines=pipelines, datasets=None, metrics=metrics, rank='f1')
print(scores)
summary = _summarize_results_datasets(scores, metrics)
print(summary)