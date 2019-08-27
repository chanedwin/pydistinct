from pydistinct.stats_estimators import *


def median_estimator(sequence):
    """
    Takes median result from all statistical estimators.

    :param sequence: sample sequence of integers
    :type sequence: array of ints
    :return: median value of all estimator
    :rtype: float
    """
    d = len(set(sequence))

    def try_wrap(func, **kwargs):
        try:
            return func(**kwargs)
        except:
            return d

    estimators = [("chao_estimator", try_wrap(chao_estimator, sequence=sequence)),
                  ("chao_lee_estimator", try_wrap(chao_lee_estimator, sequence=sequence)),
                  ("jackknife_estimator", try_wrap(jackknife_estimator, sequence=sequence)),
                  ("sichel_estimator", try_wrap(sichel_estimator, sequence=sequence)),
                  ("bootstrap_estimator", try_wrap(bootstrap_estimator, sequence=sequence)),
                  ("method_of_moments_estimator", try_wrap(method_of_moments_estimator, sequence=sequence))
                  ]

    for n_pop in [1000, 10000000]:
        estimators.append(("horvitz_thompson_estimator_{}".format(n_pop),
                           try_wrap(horvitz_thompson_estimator, sequence=sequence,
                                    pop_estimator=lambda x: x * n_pop)))
        estimators.append(("method_of_moments_v2_estimator_{}".format(n_pop),
                           try_wrap(method_of_moments_v2_estimator, sequence=sequence,
                                    pop_estimator=lambda x: x * n_pop)))

        estimators.append(("method_of_moments_v3_estimator_{}".format(n_pop),
                           try_wrap(method_of_moments_v3_estimator, sequence=sequence,
                                    pop_estimator=lambda x: x * n_pop)))

        estimators.append(("smoothed_jackknife_estimator_{}".format(n_pop),
                           try_wrap(smoothed_jackknife_estimator, sequence=sequence,
                                    pop_estimator=lambda x: x * n_pop)))

    return np.median(list(map(lambda x: x[1], estimators)))
