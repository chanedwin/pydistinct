import math
import numpy as np


def _check_iterable(sequence):
    try:
        iter(sequence)
    except TypeError:
        raise Exception("Object returned is not iterable")


def _get_attribute_counts(sequence: np.ndarray):
    """
    counts each unique attribute in a sequence

    :param sequence: observed sequence of integers
    :type sequence: list of int
    :return: dictionary with keys as attributes and values as counts of these attributes
    :rtype: dictionary (str -> int)
    """
    from itertools import groupby

    attribute_counts = {}
    for key, group in groupby(sorted(sequence), key=lambda x: x):
        attribute_counts[key] = len(list(group))

    return attribute_counts


def _get_frequency_dictionary(sequence: np.ndarray):
    """
    counts the frequency of attributes (group by count)

    :param sequence: observed sequence of integers
    :type sequence: list of int
    :return: dictionary with keys as frequencies and values as counts of these frequency
    :rtype: dict
    """
    from itertools import groupby

    attribute_counts = _get_attribute_counts(sequence)

    frequency_dictionary = {}
    for key, group in groupby(sorted(attribute_counts.items(), key=lambda x: x[1]), key=lambda x: x[1]):
        frequency_dictionary[key] = len(list(group))

    return frequency_dictionary


def _compute_birthday_problem_probability(sequence: np.ndarray, lower_bound_probability=0.1):
    """

    Suppose we draw N samples and all are distinct values. We can compute the probability of this event happening
    (N unique values in N draws) for specific R number of distinct values within a population, this helps us put a
    bound on the estimated number of distinct values.
    We estimate the number of distinct values as the population number where there is a 10% chance
    that these N distinct draws could happened due to random chance alone

    #could be optimised somehow

    :param sequence: observed sequence of integers
    :type sequence: list of int
    :param lower_bound_probability: lower bound probability for such an event happening by random chance
    :type lower_bound_probability: float (0 to 1)
    :return: estimated lower bound for distinct number of integers
    :rtype: int
    """
    print(
        "Using collision probability to compute lower bound of distinct count ie min(n) for which P(n) > 0.01")
    d = len(set(sequence))
    i = d
    while True:  # try different lower bounds
        lower_bound = d + i
        multiplied = 1
        for j in range(d):
            multiplied *= (lower_bound - j) / lower_bound
        if multiplied > lower_bound_probability:
            break
        i += 1
    return lower_bound


def memoized_gamma(x, memo_dict):
    """
    standard implementation of the gamma function

    :param x:value to evaluate gamma function at
    :type x: float
    :param memo_dict: memoized dictionary for precomputed gamma values
    :type memo_dict: dict
    :return: value of gamma function evaluated at x, and memoized results
    :rtype: int, dict
    """
    x = int(x)
    if x in memo_dict:
        return memo_dict[x], memo_dict
    else:
        result = math.lgamma(x)
        memo_dict[x] = result
        return result, memo_dict


def memoized_h_x(x, n, n_pop, memo_dict):
    """
    used in horvitz, method of moments (v1 and v2) and the smoothed jackknife estimators

    @KengHwee

    :param x:  h function evaluated at point x
    :type x: int
    :param n: length of sequence seen
    :type n: int
    :param n_pop: estimate of total number of tuples in Relation
    :type n_pop: int
    :param memo_dict: dictionary where gamma functions results are saved
    :type memo_dict: dict
    :return: value of h function evaluated at x, and memoized results
    :rtype: float, dict
    """

    gamma_num_1, memo_dict = memoized_gamma(n_pop - x + 1, memo_dict)
    gamma_num_2, memo_dict = memoized_gamma(n_pop - n + 1, memo_dict)
    gamma_denom_1, memo_dict = memoized_gamma(n_pop - x - n + 1, memo_dict)
    gamma_denom_2, memo_dict = memoized_gamma(n_pop + 1, memo_dict)

    result = np.exp(gamma_num_1 + gamma_num_2 - gamma_denom_1 - gamma_denom_2)
    return result, memo_dict


def h_x(x, n, n_pop):
    """
    used in horvitz, method of moments (v1 and v2) and the smoothed jackknife estimators

    :param x: h function evaluated at point x
    :type x: int
    :param n: length of sequence seen
    :type n: int
    :param n_pop: estimate of total number of tuples in Relation
    :type n_pop: int
    :return: value of h function evaluated at x
    :rtype: float
    """

    gamma_num_1, memo_dict = memoized_gamma(n_pop - x + 1, {})
    gamma_num_2, memo_dict = memoized_gamma(n_pop - n + 1, memo_dict)
    gamma_denom_1, memo_dict = memoized_gamma(n_pop - x - n + 1, memo_dict)
    gamma_denom_2, memo_dict = memoized_gamma(n_pop + 1, memo_dict)

    result = np.exp(gamma_num_1 + gamma_num_2 - gamma_denom_1 - gamma_denom_2)
    return result
