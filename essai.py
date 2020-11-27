import site
import sys

site.addsitedir('Orion/')
site.addsitedir('MLPrimitives/')

from orion.benchmark import benchmark

pipelines = [
    'orbit',
]

metrics = ['f1']

data = {
    'YAHOOA2': ['synthetic_85']
}

scores = benchmark(pipelines=pipelines, datasets=data, metrics=metrics, rank='f1')
print(scores)