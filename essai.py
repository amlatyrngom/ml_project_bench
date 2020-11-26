import site
import sys
from Orion.orion.evaluation import CONTEXTUAL_METRICS as METRICS
from Orion.orion.evaluation import contextual_confusion_matrix
from functools import partial

site.addsitedir('Orion/')
site.addsitedir('MLPrimitives/')

from orion.benchmark import benchmark

pipelines = [
    'mssa'
]

del METRICS['accuracy']
METRICS['confusion_matrix'] = contextual_confusion_matrix
metrics = {k: partial(fun, weighted=False) for k, fun in METRICS.items()}

data = {
    'MSL': ['A-2','A-1','A-3']
}

scores = benchmark(pipelines=pipelines, datasets=data, metrics=metrics, rank='f1')
print(scores)