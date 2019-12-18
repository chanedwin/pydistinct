from pydistinct.stats_estimators import *
from pydistinct.utils import _get_frequency_dictionary, _get_attribute_counts, _compute_birthday_problem_probability


def median_estimator(sequence):
    """
    Takes median result from faster and generally more reliable statistical estimators

    :param sequence: sample sequence of integers
    :type sequence: array of ints
    :return: median value of all estimator
    :rtype: float
    """

    n = len(sequence)
    d = len(set(sequence))
    if d == n:
        return _compute_birthday_problem_probability(sequence)
    attribute_counts = _get_attribute_counts(sequence)
    frequency_dictionary = _get_frequency_dictionary(sequence)

    cache = {"n": n, "d": d, "attr": attribute_counts, "freq": frequency_dictionary}

    def try_wrap(func, **kwargs):
        try:
            return func(**kwargs)
        except:
            return d

    estimators = [("chao_estimator", try_wrap(chao_estimator, sequence=sequence, cache=cache)),
                  ("chao_lee_estimator", try_wrap(chao_lee_estimator, sequence=sequence, cache=cache)),
                  ("jackknife_estimator", try_wrap(jackknife_estimator, sequence=sequence, cache=cache)),
                  ("bootstrap_estimator", try_wrap(bootstrap_estimator, sequence=sequence, cache=cache)),
                  ]

    for n_pop in [1000, 100000]:
        estimators.append(("horvitz_thompson_estimator_{}".format(n_pop),
                           try_wrap(horvitz_thompson_estimator, sequence=sequence,
                                    pop_estimator=lambda x: x * n_pop, cache=cache)))

        estimators.append(("smoothed_jackknife_estimator_{}".format(n_pop),
                           try_wrap(smoothed_jackknife_estimator, sequence=sequence,
                                    pop_estimator=lambda x: x * n_pop, cache=cache)))

        estimators.append(("method_of_moments_v2_estimator_{}".format(n_pop),
                           try_wrap(method_of_moments_v2_estimator, sequence=sequence,
                                    pop_estimator=lambda x: x * n_pop, cache=cache)))
    return np.median(list(map(lambda x: x[1], estimators)))


def full_median_estimator(sequence):
    """
    Takes median result from all statistical estimators. 10 times slower than normal median estimator

    :param sequence: sample sequence of integers
    :type sequence: array of ints
    :return: median value of all estimator
    :rtype: float
    """

    n = len(sequence)
    d = len(set(sequence))
    if d == n:
        return _compute_birthday_problem_probability(sequence)
    attribute_counts = _get_attribute_counts(sequence)
    frequency_dictionary = _get_frequency_dictionary(sequence)

    cache = {"n": n, "d": d, "attr": attribute_counts, "freq": frequency_dictionary}

    def try_wrap(func, **kwargs):
        try:
            return func(**kwargs)
        except:
            return d

    estimators = [("chao_estimator", try_wrap(chao_estimator, sequence=sequence, cache=cache)),
                  ("chao_lee_estimator", try_wrap(chao_lee_estimator, sequence=sequence, cache=cache)),
                  ("jackknife_estimator", try_wrap(jackknife_estimator, sequence=sequence, cache=cache)),
                  ("bootstrap_estimator", try_wrap(bootstrap_estimator, sequence=sequence, cache=cache)),
                  ("method_of_moments_estimator",
                   try_wrap(method_of_moments_estimator, sequence=sequence, cache=cache)),
                  ("sichel_estimator", try_wrap(sichel_estimator, sequence=sequence, cache=cache)),
                  ("shlosser_estimator", try_wrap(shlossers_estimator, sequence=sequence, cache=cache)),
                  ("hybrid_estimator", try_wrap(hybrid_estimator, sequence=sequence, cache=cache))]

    for n_pop in [1000, 100000]:
        estimators.append(("horvitz_thompson_estimator_{}".format(n_pop),
                           try_wrap(horvitz_thompson_estimator, sequence=sequence,
                                    pop_estimator=lambda x: x * n_pop, cache=cache)))
        estimators.append(("method_of_moments_v2_estimator_{}".format(n_pop),
                           try_wrap(method_of_moments_v2_estimator, sequence=sequence,
                                    pop_estimator=lambda x: x * n_pop, cache=cache)))

        estimators.append(("smoothed_jackknife_estimator_{}".format(n_pop),
                           try_wrap(smoothed_jackknife_estimator, sequence=sequence,
                                    pop_estimator=lambda x: x * n_pop, cache=cache)))

        estimators.append(("method_of_moments_v3_estimator_{}".format(n_pop),
                           try_wrap(method_of_moments_v3_estimator, sequence=sequence,
                                    pop_estimator=lambda x: x * n_pop, cache=cache)))

    return np.median(list(map(lambda x: x[1], estimators)))
