from src.euclidean_distance import euclidean_distance
import operator
import pandas as pd


def get_distances(training_set, test_set, number_of_dimensions):
    distances = {}
    # Calculating euclidean distance between each row of training
    # data and test data
    for i in range(len(training_set)):
        distance = euclidean_distance(test_set, training_set.iloc[i],
                                      number_of_dimensions)
        distances[i] = distance[0]

    # Sorting them on the basis of distance
    return sorted(distances.items(), key=operator.itemgetter(1))


def get_nearest_neighbors(sorted_distances, k):
    neighbors = []
    # Extracting top k neighbors
    for i in range(k):
        neighbors.append(sorted_distances[i][0])
    return neighbors


def vote_on_neighbor_class(training_set, neighbors):
    class_votes = {}

    for i in range(len(neighbors)):
        response = training_set.iloc[neighbors[i], -1]

        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1

    return sorted(class_votes.items(), key=operator.itemgetter(1),
                  reverse=True)


def knn(training_set, test_set, k):

    sorted_distances = get_distances(training_set, test_set, test_set.shape[1])
    neighbors = get_nearest_neighbors(sorted_distances, k)

    sorted_votes = vote_on_neighbor_class(training_set, neighbors)
    return sorted_votes[0][0], neighbors

