from src.knn import get_distances, get_nearest_neighbors,\
    vote_on_neighbor_class, knn
import pandas as pd


training_set = pd.DataFrame([[1, 1, 3, 1, "violet"], [1, 3, 1, 1, "violet"],
                             [1, 1, 1, 2, "violet"], [6, 1, 1, 1, "red"]])
test_set = pd.DataFrame([[1, 1, 1, 1]])


def test_distances():
    assert get_distances(training_set, test_set, test_set.shape[1]) == \
           [(2, 1.0), (0, 2.0), (1, 2.0), (3, 5.0)]


def test_get_neighbors():
    sorted_distances = get_distances(training_set, test_set, test_set.shape[1])
    assert get_nearest_neighbors(sorted_distances, 4) == [2, 0, 1, 3]


def test_vote_on_neighbor_class():
    sorted_distances = get_distances(training_set, test_set, test_set.shape[1])
    neighbors = get_nearest_neighbors(sorted_distances, 4)
    assert vote_on_neighbor_class(training_set, neighbors) == [("violet", 3),
                                                               ("red", 1)]


def test_knn():
    assert knn(training_set, test_set, 4) == ("violet",  [2, 0, 1, 3])
