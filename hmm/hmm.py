import numpy as np
import scipy.stats as st
from operator import add
from itertools import product
from scipy.misc import logsumexp


def gaussian_emission_forwards_backwards(signal, means, variances,
                                         transition_probs, starting_dist):
    """
    This implementation follows Murphy, p. 612.

    params:
    ------
    signal: an iterable of observed values
    means: an iterable of the means for each state
    variances: an iterable of the variances for each state
    transition_probs: a matrix where the value at row j column i is the
        probability of transitioning from state j to state i
    starting_dist: an iterable of the probabilities of starting at each
        state

    returns:
    ------
    gamma: a matrix where the value at row t column j is the probability of
        being in state j in timestep t
    xi: 3D array where the value at xi[t, i, j] is the joint probability of
        state i at timestep t and state j at timestep t + 1
    observed_log_likelihood: the observed data log likelihood
    """
    num_motifs = len(means)
    num_positions = len(signal)

    # suppress warning for 0 valued transition probabilities
    with np.errstate(divide='ignore'):
        log_start = np.log(starting_dist)
        log_tp = np.log(transition_probs)

    # probability of each timestep under each distribution
    probs = np.ones((num_positions, num_motifs))
    for t, s in enumerate(signal):
        probs[t] = st.norm.logpdf(s, means, variances)

    # forwards
    alpha = np.ones((num_positions, num_motifs))
    alpha[0] = probs[0] + log_start
    for t in range(1, num_positions):
        alpha[t] = probs[t] + logsumexp(log_tp.T + alpha[t - 1], axis=1)

    # backwards
    beta = np.zeros((num_positions, num_motifs))
    for t in range(num_positions - 2, -1, -1):
        beta[t] = logsumexp(log_tp + (beta[t + 1] + probs[t + 1]), axis=1)

    # compute and normalize gamma
    gamma = alpha + beta
    for t in range(num_positions):
        gamma[t] = gamma[t] - logsumexp(gamma[t])

    # compute and normalize xi
    xi = np.zeros((num_positions - 1, num_motifs, num_motifs))
    for t in range(num_positions - 1):
        xi[t] = log_tp + np.array([
            add(*p) for p in product(alpha[t], probs[t + 1] + beta[t + 1])
        ]).reshape(num_motifs, num_motifs)
        xi[t] = xi[t] - logsumexp(xi[t])

    observed_log_likelihood = logsumexp(alpha[num_positions - 1])
    return np.exp(gamma), np.exp(xi), observed_log_likelihood
