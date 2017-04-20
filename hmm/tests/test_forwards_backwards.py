import numpy as np
from hmm import gaussian_emission_forwards_backwards


def test_gaussian_emission_forwards_backwards():
    # given
    signal = [0, 2, 3, 2, 1, 2, 2, 1, 3, 2]
    means = list(range(4))
    variances = [0.01] * 4
    transition_probs = np.array([[.25] * 4] * 4)

    # when
    result = gaussian_emission_forwards_backwards(signal, means, variances,
                                                  transition_probs)

    expected = np.array([
        [1.,  0.,  0.,  0.],
        [0.,  0.,  1.,  0.],
        [0.,  0.,  0.,  1.],
        [0.,  0.,  1.,  0.],
        [0.,  1.,  0.,  0.],
        [0.,  0.,  1.,  0.],
        [0.,  0.,  1.,  0.],
        [0.,  1.,  0.,  0.],
        [0.,  0.,  0.,  1.],
        [0.,  0.,  1.,  0.],
    ])

    # then
    assert (result == expected).all()
