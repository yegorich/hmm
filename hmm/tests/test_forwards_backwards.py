import numpy as np
from hmm import gaussian_emission_forwards_backwards


def test_gaussian_emission_forwards_backwards():
    # given
    signal = [0, 2, 3, 2, 1, 2, 2, 1, 3, 2]
    starting_dist = [.1, .4, .4, .1]
    means = list(range(4))
    variances = [0.01] * 4
    transition_probs = np.array([[.25] * 4] * 4)

    # when
    result = gaussian_emission_forwards_backwards(
        signal, means, variances, transition_probs, starting_dist)

    expected = np.array([
        [1., 0., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
        [0., 0., 1., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 1., 0.],
        [0., 1., 0., 0.],
        [0., 0., 0., 1.],
        [0., 0., 1., 0.],
    ])

    # then
    assert (result[0] == expected).all()
    assert ((result[1] == 0) + (result[1] == 1)).all()
