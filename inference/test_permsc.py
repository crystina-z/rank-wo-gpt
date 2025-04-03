import glob
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm
from nirtools.ir import load_runs, write_runs

from permsc import KemenyOptimalAggregator, sum_kendall_tau, ranks_from_preferences


TOPK = 20


def test_basic():
    preferences = np.array([[1, 2, 0, 3], [1, 2, 3, 0], [1, 2, 0, 4]])
    ranks = ranks_from_preferences(preferences)
    y_optimal = KemenyOptimalAggregator().aggregate(preferences)
    y_optimal_ranks = ranks_from_preferences(y_optimal)
    print(y_optimal)  # [1, 2, 0, 3]
    print(sum_kendall_tau(ranks, y_optimal_ranks))  # the sum of the Kendall tau distances


@dataclass
class IndexOrganizer:
    doc_ids: list # list of actual document ids

    def __post_init__(self):
        if isinstance(self.doc_ids[0], list): # flatten the list if it is nested
            self.doc_ids = [item for sublist in self.doc_ids for item in sublist]

        self.unique_doc_ids = unique_doc_ids = sorted(set(self.doc_ids))
        self.doc_id_to_int_id = {doc_id: i for i, doc_id in enumerate(unique_doc_ids)}
        self.int_id_to_doc_id = {i: doc_id for i, doc_id in enumerate(unique_doc_ids)}
    
    def to_int_id(self, doc_id: str) -> int:
        return self.doc_id_to_int_id[doc_id]
    
    def to_doc_id(self, int_id: int) -> str:
        return self.int_id_to_doc_id[int_id]

    def convert_docIds_to_intIds(self, doc_ids: list, add_missing: bool = True) -> list:
        if add_missing:
            missing_docs = list(set(self.unique_doc_ids) - set(doc_ids))
            doc_ids += missing_docs

        return [self.to_int_id(doc_id) for doc_id in doc_ids]

    def convert_intIds_to_docIds(self, int_ids: list) -> list:
        return [self.to_doc_id(int_id) for int_id in int_ids]



def aggregate_doc_ids(list_of_docids):
    """
    list_of_docids: all list of sorted document ids that will be aggregated
    """
    id_organizer = IndexOrganizer(doc_ids=list_of_docids)
    list_of_int_ids = [id_organizer.convert_docIds_to_intIds(doc_ids) for doc_ids in list_of_docids]

    preferences = np.array(list_of_int_ids)
    print(f"Preferences shape: {preferences.shape}")
    print(f"Number of unique documents: {len(id_organizer.unique_doc_ids)}")

    y_optimal = KemenyOptimalAggregator().aggregate(preferences)
    optimal_doc_ids = id_organizer.convert_intIds_to_docIds(y_optimal)

    # ranks = ranks_from_preferences(preferences)
    # y_optimal_ranks = ranks_from_preferences(y_optimal)
    # print(sum_kendall_tau(ranks, y_optimal_ranks))  # the sum of the Kendall tau distances

    return optimal_doc_ids


def load_runfiles(runfile_pattern: str):
    """
    runfile_pattern: pattern of the runfile to be loaded
    """
    def get_sorted_docids(docid2score: dict):
        # rank according to relevance score, from high to low
        return sorted(docid2score.keys(), key=lambda x: docid2score[x], reverse=True)

    runfiles = glob.glob(runfile_pattern)
    runs = [load_runs(runfile, topk=TOPK) for runfile in runfiles]
    qids = list(runs[0].keys())
    aggregated_runs = {
        qid: [get_sorted_docids(run[qid]) for run in runs] for qid in qids
    }
    return aggregated_runs


def main():
    pattern = "rerank-results/Qwen.Qwen2.5-7B-Instruct/window-20-step-10/trec-dl-2019/rank-wo-gpt-Seed-*-Temp-0.25.trec"
    initial_runs = load_runfiles(pattern)

    psc_runs = {}
    for qid in tqdm(initial_runs, desc="Aggregating runs"):
        optimal_doc_ids = aggregate_doc_ids(initial_runs[qid])
        psc_runs[qid] = {
            doc_id: - rank for rank, doc_id in enumerate(optimal_doc_ids) # score: -rank; smaller the rank, higher the score
        }

    write_runs(psc_runs, f"test-psc-2019.{TOPK}.trec")



if __name__ == "__main__":
    # test_basic()
    main()
