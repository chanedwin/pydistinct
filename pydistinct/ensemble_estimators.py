from pydistinct.stats_estimators import *
from pydistinct.utils import _compute_birthday_problem_probability


def median_estimator(sequence=None, attributes=None):
    """
    Takes median result from faster and generally more reliable statistical estimators

    :param sequence: sample sequence of integers
    :type sequence: array of ints
    :param attributes: dictionary with keys as the unique elements and values as
                        counts of those elements
    :type attributes: dictionary where keys can be any type, values must be integers
    :return: median value of all estimator
    :rtype: float
    """

    if sequence is None and attributes is None:
        raise Exception("Must provide a sequence, or a dictionary of attribute counts ")

    if sequence is not None:
        n, d, attribute_counts, frequency_dictionary = precompute_from_seq(sequence)
    else:
        n, d, attribute_counts, frequency_dictionary = precompute_from_attr(attributes)

    if n == d:
        return _compute_birthday_problem_probability(d)

    cache = {"n": n, "d": d, "attr": attribute_counts, "freq": frequency_dictionary}

    def try_wrap(func, **kwargs):
        try:
            return func(**kwargs)
        except:
            return d

    estimators = [("chao_estimator", try_wrap(chao_estimator, cache=cache)),
                  ("chao_lee_estimator", try_wrap(chao_lee_estimator, cache=cache)),
                  ("jackknife_estimator", try_wrap(jackknife_estimator, cache=cache)),
                  ("bootstrap_estimator", try_wrap(bootstrap_estimator, cache=cache)),
                  ]

    for n_pop in [1000, 100000]:
        estimators.append(("horvitz_thompson_estimator_{}".format(n_pop),
                           try_wrap(horvitz_thompson_estimator,
                                    pop_estimator=lambda x: x * n_pop, cache=cache)))

        estimators.append(("smoothed_jackknife_estimator_{}".format(n_pop),
                           try_wrap(smoothed_jackknife_estimator,
                                    pop_estimator=lambda x: x * n_pop, cache=cache)))

        estimators.append(("method_of_moments_v2_estimator_{}".format(n_pop),
                           try_wrap(method_of_moments_v2_estimator,
                                    pop_estimator=lambda x: x * n_pop, cache=cache)))
    return np.median(list(map(lambda x: x[1], estimators)))


def full_median_estimator(sequence=None, attributes=None):
    """
    Takes median result from all statistical estimators. 10 times slower than normal median estimator

    :param sequence: sample sequence of integers
    :type sequence: array of ints
    :param attributes: dictionary with keys as the unique elements and values as
                        counts of those elements
    :type attributes: dictionary where keys can be any type, values must be integers
    :return: median value of all estimator
    :rtype: float
    """

    if sequence is None and attributes is None:
        raise Exception("Must provide a sequence, or a dictionary of attribute counts ")

    if sequence is not None:
        n, d, attribute_counts, frequency_dictionary = precompute_from_seq(sequence)
    else:
        n, d, attribute_counts, frequency_dictionary = precompute_from_attr(attributes)

    if n == d:
        return _compute_birthday_problem_probability(d)

    cache = {"n": n, "d": d, "attr": attribute_counts, "freq": frequency_dictionary}

    def try_wrap(func, **kwargs):
        try:
            return func(**kwargs)
        except:
            return d

    estimators = [("chao_estimator", try_wrap(chao_estimator, cache=cache)),
                  ("chao_lee_estimator", try_wrap(chao_lee_estimator, cache=cache)),
                  ("jackknife_estimator", try_wrap(jackknife_estimator, cache=cache)),
                  ("bootstrap_estimator", try_wrap(bootstrap_estimator, cache=cache)),
                  ("method_of_moments_estimator",
                   try_wrap(method_of_moments_estimator, cache=cache)),
                  ("sichel_estimator", try_wrap(sichel_estimator, cache=cache)),
                  ("shlosser_estimator", try_wrap(shlossers_estimator, cache=cache)),
                  ("hybrid_estimator", try_wrap(hybrid_estimator, cache=cache))]

    for n_pop in [1000, 100000]:
        estimators.append(("horvitz_thompson_estimator_{}".format(n_pop),
                           try_wrap(horvitz_thompson_estimator,
                                    pop_estimator=lambda x: x * n_pop, cache=cache)))
        estimators.append(("method_of_moments_v2_estimator_{}".format(n_pop),
                           try_wrap(method_of_moments_v2_estimator,
                                    pop_estimator=lambda x: x * n_pop, cache=cache)))

        estimators.append(("smoothed_jackknife_estimator_{}".format(n_pop),
                           try_wrap(smoothed_jackknife_estimator,
                                    pop_estimator=lambda x: x * n_pop, cache=cache)))

        estimators.append(("method_of_moments_v3_estimator_{}".format(n_pop),
                           try_wrap(method_of_moments_v3_estimator,
                                    pop_estimator=lambda x: x * n_pop, cache=cache)))

    return np.median(list(map(lambda x: x[1], estimators)))
