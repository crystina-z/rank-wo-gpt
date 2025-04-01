import ir_measures
from ir_measures import *
import ir_datasets

from nirtools.ir import load_runs

from pprint import pprint


METRICS = [nDCG@10, R@100]

def evaluate(runfile, dataset_name):
    reranked_runs = load_runs(runfile)
    qrels = ir_datasets.load(dataset_name).qrels_iter()
    results = ir_measures.calc_aggregate(METRICS, qrels, reranked_runs)

    print(f"Evaluating file {runfile} on dataset {dataset_name}:")
    pprint(results)
