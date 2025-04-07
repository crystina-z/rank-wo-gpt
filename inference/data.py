import os
import ir_datasets
from nirtools.ir import load_runs

SUPPORTED_DATASETS = [
    "msmarco-passage/trec-dl-2019",
    "msmarco-passage/trec-dl-2020",    
]


def load_data(dataset, topk=100): 
    assert dataset in SUPPORTED_DATASETS, f"Dataset {dataset} not supported"

    if "trec-dl" in dataset:
        ds = ir_datasets.load(f"{dataset}/judged")
    else:
        ds = ir_datasets.load(f"{dataset}")


    DATA_PATH = os.path.dirname(os.path.abspath(__file__))
    runfile = f"{DATA_PATH}/data/{dataset}/bm25.trec"

    runs = load_runs(runfile, topk=topk)
    print("Loaded runfile")

    qid2query = {q.query_id: q.text for q in ds.queries_iter()}
    docid2doc = {doc.doc_id: doc.text for doc in ds.docs_iter()}
    print("Loaded docs")

    return runs, qid2query, docid2doc
