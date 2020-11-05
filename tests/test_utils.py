"""Test for utils."""
import numpy
import pandas as pd
import scipy.spatial.distance

from nlkda.utils import all_knn, get_skyline


def test_all_knn():
    """Test all_knn."""
    n = 13
    k = 3
    d = 2
    x = numpy.random.uniform(size=(n, d))
    dist = all_knn(
        x=x,
        k=k,
        batch_size=n // 2,
        distance="minkowski",
    )
    exp_dist = scipy.spatial.distance.cdist(x, x)
    exp_dist = numpy.sort(exp_dist, axis=-1)[:, :k]
    numpy.testing.assert_almost_equal(dist, exp_dist)


def test_get_skyline():
    df = pd.DataFrame({"col1": [1, 1.2, 3, 4, 7], "col2": [1, .5, 6, 2, 5], "col3": [3, 5, .3, 6, 2]})
    skyline_two_smaller_is_better = get_skyline(selection=df, columns=["col1", "col2"], smaller_is_better=(True, True))
    skyline_three_smaller_is_better = get_skyline(selection=df, columns=["col1", "col2", "col3"],
                                                  smaller_is_better=(True, True, True))
    skyline_three_larger_is_better = get_skyline(selection=df, columns=["col1", "col2", "col3"],
                                                 smaller_is_better=(False, False, False))

    assert (list(skyline_three_smaller_is_better.index.values) == [0, 1, 2, 4])
    assert (list(skyline_three_larger_is_better.index.values) == [2, 3, 4])
    assert (list(skyline_two_smaller_is_better.index.values) == [0, 1])