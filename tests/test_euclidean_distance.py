from src.euclidean_distance import euclidean_distance


def test_2_dimensional_distance():
    assert euclidean_distance([1, 1], [1, 3], 2) == 2
