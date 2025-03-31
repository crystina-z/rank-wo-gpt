import numpy as np
import permsc
print(permsc.__file__)

from permsc import KemenyOptimalAggregator, sum_kendall_tau, ranks_from_preferences

def test_basic():
    preferences = np.array([[1, 2, 0, 3], [1, 2, 3, 0], [1, 2, 0, 3]])
    ranks = ranks_from_preferences(preferences)

    y_optimal = KemenyOptimalAggregator().aggregate(preferences)
    y_optimal_ranks = ranks_from_preferences(y_optimal)
    print(y_optimal)  # [1, 2, 0, 3]
    print(sum_kendall_tau(ranks, y_optimal_ranks))  # the sum of the Kendall tau distances


def aggregate_documents(list_of_docids):
    pass


def main():
    test_basic()


if __name__ == "__main__":
    main()