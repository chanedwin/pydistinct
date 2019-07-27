import math

import numpy as np
from scipy.optimize import broyden1, broyden2
from scipy.stats import chi2

from pydistinct.utils import _get_attribute_counts, _get_frequency_dictionary, _compute_birthday_problem_probability, \
    h_x, \
    memoized_h_x, _check_iterable


def goodmans_estimator(sequence):
    """
    Implementation of goodmans estimator from Goodman 1949 : throws an error if N is too high due to
    numerical complexity

    :param sequence: sample sequence of integers
    :type sequence: array of ints
    :return: estimated distinct count
    :rtype: int
    """
    _check_iterable(sequence)

    n = len(sequence)
    d = len(set(sequence))
    N = n * 2
    frequency_dictionary = _get_frequency_dictionary(sequence)

    if d == n:
        print("f2 frequency is 0, Chao estimator returns infinite results")
        return _compute_birthday_problem_probability(sequence)

    memo_fact_dict = {}

    def memo_fact(x, memo_dict):
        if x not in memo_dict:
            memo_dict[x] = np.math.factorial(x)
        return memo_dict[x], memo_dict

    sum_goodman = 0
    for i in frequency_dictionary.keys():
        num_1, memo_fact_dict = memo_fact(N - n + i - 1, memo_fact_dict)
        num_2, memo_fact_dict = memo_fact(n - i, memo_fact_dict)
        denom_1, memo_fact_dict = memo_fact(N - n - 1, memo_fact_dict)
        denom_2, memo_fact_dict = memo_fact(n, memo_fact_dict)
        sum_goodman += ((-1) ** (i + 1)) * num_1 * num_2 * frequency_dictionary[i] / (denom_1 * denom_2)
    d_goodman = d + sum_goodman
    return d_goodman


def chao_estimator(sequence):
    """

    Implementation of Chao's estimator from Chao 1984, using counts of values that appear exactly once and twice

    d_chao = d + (f_1)^2/(2*(f_2))
    returns birthday problem solution if there are no sequences observed of frequency 2
    (ie each distinct value observed is never seen again)

    also makes insane bets (10x) when every point observed is almost unique. could be good or bad

    :param sequence: sample sequence of integers
    :type sequence: array of ints
    :return: estimated distinct count
    :rtype: int
    """
    _check_iterable(sequence)

    frequency_dictionary = _get_frequency_dictionary(sequence)
    f1 = frequency_dictionary[1]

    if 2 not in frequency_dictionary:
        print("f2 frequency is 0, Chao estimator returns infinite results")
        return _compute_birthday_problem_probability(sequence)

    f2 = frequency_dictionary[2]
    d_chao = len(set(sequence)) + np.square(f1) / (2 * f2)

    return d_chao


def jackknife_estimator(sequence):
    """
    Jackknife scheme for estimating D (Ozsoyoglu et al., 1991)
    good at regimes where sample size is close to actual number of points

    D^hat_c_j = d_n - (n - l)(d_(n-1)- d_n).

    :param sequence: sample sequence of integers
    :type sequence: array of ints
    :return: estimated distinct count
    :rtype: int
    """
    _check_iterable(sequence)

    n = len(sequence)
    d = len(set(sequence))
    attribute_counts = _get_attribute_counts(sequence)

    if d == n:
        print("f2 frequency is 0, Chao estimator returns infinite results")
        return _compute_birthday_problem_probability(sequence)

    sum_d_n_minus_1 = 0
    for i in range(n):
        tuple_k = sequence[i]
        if attribute_counts[tuple_k] == 1:
            sum_d_n_minus_1 += d - 1
        else:
            sum_d_n_minus_1 += d
    d_n_minus_1 = sum_d_n_minus_1 / n

    d_jackknife = d - (n - 1) * (d_n_minus_1 - d)

    return d_jackknife


def chao_lee_estimator(sequence):
    """
    Implementation of Chao and Lee's estimator (Chao and Lee, 1984) using a natural estimator of coverage

    gamma hat is an estimator for the squared coefficient of variation of the frequencies

    :param sequence: sample sequence of integers
    :type sequence: array of ints
    :return: estimated distinct count
    :rtype: int
    """
    _check_iterable(sequence)

    n = len(sequence)
    d = len(set(sequence))
    frequency_dictionary = _get_frequency_dictionary(sequence)
    f1 = frequency_dictionary[1]
    attribute_counts = _get_attribute_counts(sequence)

    if d == n:
        print("f2 frequency is 0, Chao estimator returns infinite results")
        return _compute_birthday_problem_probability(sequence)

    c_hat = 1 - f1 / n  # throws an error if f1 = n (all values observed are unique.)
    if c_hat == 0:
        print("all values observed are unique, using birthday problem approach")
        return _compute_birthday_problem_probability(sequence)
    gamma_hat_squared = (1 / d) * np.var(list(attribute_counts.values())) / ((n / d) ** 2)

    d_cl = (d / c_hat) + ((n * (1 - c_hat) * gamma_hat_squared) / c_hat)

    return d_cl


def shlossers_estimator(sequence, pop_estimator=lambda x: x * 2, n_pop=None):
    """
    Implementation of Shlosser's Estimator (Shlosser 1981) using a Bernoulli Sampling scheme

    Note : Hard to determine q (probability of being included)

    :param sequence: sample sequence of integers
    :type sequence: array of ints
    :param pop_estimator: function to estimate population size if possible
    :type pop_estimator: function that takes in the length of sequence (int) and outputs
                         the estimated population size (int)
    :param n_pop: estimate of population size if available, will be used over pop_estimator function
    :type n_pop: int
    :return: estimated distinct count
    :rtype: int
    """
    _check_iterable(sequence)

    n = len(sequence)
    d = len(set(sequence))

    if d == n:
        print("f2 frequency is 0, Chao estimator returns infinite results")
        return _compute_birthday_problem_probability(sequence)

    frequency_dictionary = _get_frequency_dictionary(sequence)
    if not n_pop:
        n_pop = pop_estimator(n)

    f1 = frequency_dictionary[1]
    q = n / n_pop  # placeholder q determination - TODO
    numerator = 0
    denominator = 0
    for i in range(1, n):
        if i in frequency_dictionary:
            result = frequency_dictionary[i]
        else:
            result = 0
        numerator += ((1 - q) ** i) * result
        denominator += (i * q * (1 - q) ** (i - 1)) * result

    d_shlosser = d + f1 * numerator / denominator
    return d_shlosser


def sichel_estimator(sequence):
    """
    Implementation of Sichel’s Parametric Estimator (Sichel 1986a, 1986b and 1992)
    which uses a zero-truncated generalized inverse Gaussian-Poisson to estimate D

    implementation uses broyden 2 to solve and search linear space for good solution

    :param sequence: sample sequence of integers
    :type sequence: array of ints
    :return: estimated distinct count
    :rtype: int
    """
    _check_iterable(sequence)

    n = len(sequence)
    d = len(set(sequence))

    if d == n:
        print("f2 frequency is 0, Chao estimator returns infinite results")
        return _compute_birthday_problem_probability(sequence)

    frequency_dictionary = _get_frequency_dictionary(sequence)

    f1 = frequency_dictionary[1]
    a = ((2 * n) / d) - np.log(n / f1)  # does not depend on g
    b = ((2 * f1) / d) + np.log(n / f1)  # does not depend on g

    def diff_eqn(g):
        result = (1 + g) * np.log(g) - a * g + b
        return result

    d_sichel_set = set()

    for value in np.linspace((f1 / n) + 0.00001, 0.999999, 20):
        try:
            g = broyden2(diff_eqn, value)
            if 1 > g > (f1 / n) and ((n * g) / f1) > 0:
                b_hat = g * np.log((n * g) / f1) / (1 - g)
                c_hat = (1 - g ** 2) / (n * (g ** 2))
                d_sichel = 2 / (b_hat * c_hat)
                d_sichel_set.add(d_sichel)
        except Exception as e:
            continue

    if not d_sichel_set:
        return d
    else:
        return min(d_sichel_set)


def method_of_moments_estimator(sequence):
    """
    Simple Method-of-Moments Estimator to estimate D (Haas et al, 1995)
    can be optimised (training rate, stopping value)

    d = d_moments(l -  e^(-n))/d_moments)

    solve for d_moments in d = d_moments(l -  e^(-n))/d_moments)

    :param sequence: sample sequence of integers
    :type sequence: array of ints
    :return: estimated distinct count
    :rtype: int
    """
    _check_iterable(sequence)

    n = len(sequence)
    d = len(set(sequence))

    if d == n:
        print("f2 frequency is 0, Chao estimator returns infinite results")
        return _compute_birthday_problem_probability(sequence)

    def diff_eqn(D):
        return D * (1 - np.exp((-n / D))) - d

    try:
        d_moments_1 = float(broyden1(diff_eqn, d))
    except Exception as e:
        print(e)
        d_moments_1 = 1000000000000000000000000

    try:
        d_moments_2 = float(broyden2(diff_eqn, d))
    except:
        d_moments_2 = 1000000000000000000000000

    result = min(d_moments_1, d_moments_2)
    if result == 1000000000000000000000000:
        return d
    else:
        return result


def bootstrap_estimator(sequence):
    """
    Implementation of a bootstrap estimator to estimate D (Smith and Van Bell 1984; Haas et al, 1995)

    DBoot = d + sigma (1-nj/n)^n

    :param sequence: sample sequence of integers
    :type sequence: array of ints
    :return: estimated distinct count
    :rtype: int
    """
    _check_iterable(sequence)

    n = len(sequence)
    d = len(set(sequence))

    if d == n:
        print("f2 frequency is 0, Chao estimator returns infinite results")
        return _compute_birthday_problem_probability(sequence)

    attribute_counts = _get_attribute_counts(sequence)
    bootstrap_sum = 0
    for j, n_j in attribute_counts.items():
        final_val = (1 - n_j / n) ** n
        bootstrap_sum += final_val
    d_bootstrap = d + bootstrap_sum
    return d_bootstrap


def horvitz_thompson_estimator(sequence, pop_estimator=lambda x: x * 10000000, n_pop=None):
    """
    Implementation of the Horvitz-Thompson Estimator to estimate D
    (Sarndal, Swensson, and Wretman 1992; Haas et al, 1995)

    n_j = attribute count of value j

    :param sequence: sample sequence of integers
    :type sequence: array of ints
    :param pop_estimator: function to estimate population size if possible
    :type pop_estimator: function that takes in the length of sequence (int) and outputs
                         the estimated population size (int)
    :param n_pop: estimate of population size if available, will be used over pop_estimator function
    :type n_pop: int
    :return: estimated distinct count
    :rtype: int
    """
    _check_iterable(sequence)

    n = len(sequence)
    d = len(set(sequence))

    if d == n:
        print("f2 frequency is 0, Chao estimator returns infinite results")
        return _compute_birthday_problem_probability(sequence)

    if not n_pop:
        n_pop = pop_estimator(n)
    attribute_counts = _get_attribute_counts(sequence)

    d_horvitz_thompson = 0
    memo_dict_instance = {}
    for j, n_j in attribute_counts.items():
        n_j_hat = (n_j * n_pop) / n
        h_n_j_hat, memo_dict_instance = memoized_h_x(n_j_hat, n, n_pop, memo_dict_instance)
        d_horvitz_thompson += (1 / (1 - h_n_j_hat))

    return d_horvitz_thompson


def method_of_moments_v2_estimator(sequence, pop_estimator=lambda x: x * 1000000, n_pop=None):
    """
    Method-of-Moments Estimator with equal frequency assumption while still sampling
     from a finite relation (Haas et al, 1995)

    :param sequence: sample sequence of integers
    :type sequence: array of ints
    :param pop_estimator: function to estimate population size if possible
    :type pop_estimator: function that takes in the length of sequence (int) and outputs
                         the estimated population size (int)
    :param n_pop: estimate of population size if available, will be used over pop_estimator function
    :type n_pop: int
    :return: estimated distinct count
    :rtype: int
    """
    _check_iterable(sequence)

    n = len(sequence)
    if not n_pop:
        n_pop = pop_estimator(n)
    d = len(set(sequence))

    if d == n:
        print("f2 frequency is 0, Chao estimator returns infinite results")
        return _compute_birthday_problem_probability(sequence)

    # need to implement gamma memoized function and h_x again here to enable use of global variables in broydens

    memo_dict = {}

    def memoized_gamma(x):
        """

        :param x:value to evaluate gamma function at
        :type x: float
        :param memo_dict: memoized dictionary for precomputed gamma values
        :type memo_dict: dict
        :return: value of gamma function evaluated at x, and memoized results
        :rtype: int, dict
        """
        x = int(x)
        if x in memo_dict:
            return memo_dict[x]
        else:
            result = math.lgamma(x)
            memo_dict[x] = result
            return result

    def h_x(x, n, n_pop):
        """

        :param x: h function evaluated at point x
        :type x: int
        :param n: length of sequence seen
        :type n: int
        :param n_pop: estimate of total number of tuples in Relation
        :type n_pop: int
        :return: value of h function evaluated at x
        :rtype: float
        """

        gamma_num_1 = memoized_gamma(n_pop - x + 1)
        gamma_num_2 = memoized_gamma(n_pop - n + 1)
        gamma_denom_1 = memoized_gamma(n_pop - x - n + 1)
        gamma_denom_2 = memoized_gamma(n_pop + 1)

        result = np.exp(gamma_num_1 + gamma_num_2 - gamma_denom_1 - gamma_denom_2)
        return result

    def diff_eqn(D):
        return D * (1 - h_x((n_pop / D), n, n_pop)) - d

    try:
        d_moments_1 = float(broyden1(diff_eqn, d))
    except Exception as e:
        print(e)
        d_moments_1 = 1000000000000000000000000

    try:
        d_moments_2 = float(broyden2(diff_eqn, d))
    except:
        d_moments_2 = 1000000000000000000000000

    result = min(d_moments_1, d_moments_2)
    if result == 1000000000000000000000000:
        return d
    else:
        return result


def method_of_moments_v3_estimator(sequence, pop_estimator=lambda x: x * 10000000, n_pop=None):
    """
    Method-of-Moments Estimator without equal frequency assumption (Haas et al, 1995)

    :param sequence: sample sequence of integers
    :type sequence: array of ints
    :param pop_estimator: function to estimate population size if possible
    :type pop_estimator: function that takes in the length of sequence (int) and outputs
                         the estimated population size (int)
    :param n_pop: estimate of population size if available, will be used over pop_estimator function
    :type n_pop: int
    :return: estimated distinct count
    :rtype: int
    """
    _check_iterable(sequence)

    n = len(sequence)
    if not n_pop:
        n_pop = pop_estimator(n)
    d = len(set(sequence))

    if d == n:
        print("f2 frequency is 0, Chao estimator returns infinite results")
        return _compute_birthday_problem_probability(sequence)

    attribute_counts = _get_attribute_counts(sequence)

    gamma_hat_squared = (1 / d) * np.var(list(attribute_counts.values())) / ((n_pop / d) ** 2)

    def compute_g_n(_x, _n, _n_pop):
        """
        See implementation paper for details

        :param _x: g function evaluated at point x
        :type _x: int
        :param _n: length of sequence seen
        :type _n: int
        :param _n_pop: estimate of total number of tuples in relation
        :type _n_pop: int
        :return: value of g function evaluated at x
        :rtype: float
        """
        return_sum = 0
        for k in range(_n):
            return_sum += 1 / (_n_pop - _x - _n + k)
        return return_sum

    d_mm_1 = method_of_moments_v2_estimator(sequence)
    n_pop_tilda = n_pop / d_mm_1
    d_mm_2_huge_term = 0.5 * (n_pop_tilda ** 2) * gamma_hat_squared * d_mm_1 * h_x(n_pop_tilda, n, n_pop) * (
            compute_g_n(n_pop_tilda, n, n_pop) - compute_g_n(n_pop_tilda, n, n_pop) ** 2)
    d_moments_v3 = d * (1 - h_x(n_pop_tilda, n, n_pop) + d_mm_2_huge_term) ** -1
    return d_moments_v3


def smoothed_jackknife_estimator(sequence, pop_estimator=lambda x: x * 10000000, n_pop=None):
    """
    Jackknife scheme for estimating D that accounts for true bias structures (Haas et al, 1995)

    :param sequence: sample sequence of integers
    :type sequence: array of ints
    :param pop_estimator: function to estimate population size if possible
    :type pop_estimator: function that takes in the length of sequence (int) and outputs
                         the estimated population size (int)
    :param n_pop: estimate of population size if available, will be used over pop_estimator function
    :type n_pop: int
    :return: estimated distinct count
    :rtype: int
    """
    _check_iterable(sequence)

    n = len(sequence)
    if not n_pop:
        n_pop = pop_estimator(n)
    d = len(set(sequence))

    if d == len(sequence):
        print("all values seen were unique, reverting to inverse birthday problem solution")
        return _compute_birthday_problem_probability(sequence)
    frequency_dictionary = _get_frequency_dictionary(sequence)
    attribute_counts = _get_attribute_counts(sequence)

    gamma_hat_squared = (1 / d) * np.var(list(attribute_counts.values())) / ((n_pop / d) ** 2)

    def compute_g_n_minus_one(x, n, n_pop):
        return_sum = 0
        for k in range(n - 1):
            return_sum += 1 / (n_pop - x - n + k)
        return return_sum

    d_n = d
    d_sjk_zero_hat = (d_n - frequency_dictionary[1] / n) * (
            1 - (n_pop - n + 1) * frequency_dictionary[1] / (n * n_pop)) ** -1
    n_pop_tilda = n_pop / d_sjk_zero_hat
    d_sjk = (1 - (n_pop - n_pop_tilda - n + 1) * frequency_dictionary[1] / (n * n_pop)) ** -1 * (
            d_n + n_pop * h_x(n_pop_tilda, n, n_pop) * compute_g_n_minus_one(n_pop_tilda, n,
                                                                             n_pop) * gamma_hat_squared * d_sjk_zero_hat)
    return d_sjk


def hybrid_estimator(sequence, pop_estimator=lambda x: x * 10000000, n_pop=None):
    """
    hybrid_estimator : Hybrid Estimator that uses Shlosser's estimator when data is skewed and Smooth jackknife
    estimator when data is not. Skew is computed by using an approximate chi square test for uniformity

    :param sequence: sample sequence of integers
    :type sequence: array of ints
    :param pop_estimator: function to estimate population size if possible
    :type pop_estimator: function that takes in the length of sequence (int) and outputs
                         the estimated population size (int)
    :param n_pop: estimate of population size if available, will be used over pop_estimator function
    :type n_pop: int
    :return: estimated distinct count
    :rtype: int
    """
    _check_iterable(sequence)

    n = len(sequence)
    if not n_pop:
        n_pop = pop_estimator(n)
    d = len(set(sequence))

    if d == n:
        print("f2 frequency is 0, Chao estimator returns infinite results")
        return _compute_birthday_problem_probability(sequence)

    n_bar = n / d
    attribute_counts = _get_attribute_counts(sequence)
    mu = sum((((i - n_bar) ** 2) / n_bar) for i in attribute_counts.values())
    chi_critical = chi2.isf(0.975, n - 1, loc=n_bar, scale=n_bar)  # set alpha is 0.975
    if mu <= chi_critical:
        return smoothed_jackknife_estimator(sequence, pop_estimator=pop_estimator, n_pop=n_pop)
    else:
        return shlossers_estimator(sequence, n_pop=n_pop)
