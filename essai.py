import site
import sys

site.addsitedir('Orion/')
site.addsitedir('MLPrimitives/')

from orion.benchmark import benchmark

pipelines = [
    'mssa'
]

metrics = ['f1', 'accuracy', 'recall', 'precision']

data = {
    'YAHOOA2': ['synthetic_85']
}

scores = benchmark(pipelines=pipelines, datasets=["YAHOOA2"], metrics=metrics, rank='f1')
print(scores)