import numpy as np


def euclidean_distance(data_point1, data_point2, number_of_dimensions):
    distance = 0
    for i in range(number_of_dimensions):
        distance += np.square(data_point1[i] - data_point2[i])
    return np.sqrt(distance)
