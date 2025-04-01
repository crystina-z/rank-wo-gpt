import ir_measures
from ir_measures import *
from nirtools.ir import load_runs

from pprint import pprint


METRICS = [nDCG@10, R@100]

def evaluate(runfile, dataset_name):
    reranked_runs = load_runs(runfile)
    results = ir_measures.calc_aggregate(METRICS, dataset_name, reranked_runs)

    print(f"Evaluating file {runfile} on dataset {dataset_name}:")
    pprint(results)
