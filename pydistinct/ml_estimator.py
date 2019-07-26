"""
This is a module to be used as a reference for building other modules
"""

from scipy.stats import skew, kurtosis
from statsmodels import robust
from xgboost import XGBRegressor

from pydistinct.stats_estimators import *
from pydistinct.utils import _get_attribute_counts, _get_frequency_dictionary


class XGBEstimator():
    def __init__(self):
        self.regressor = XGBRegressor()

    def fit(self, X, y, transform=True, *args, **kwargs):
        if transform:
            X = np.asarray([est_transform(i) for i in X])

        self.regressor.fit(X, y, *args, **kwargs)

    def load_pretrained_model(self,path):
        """
        get pretrained model

        :return: pretrained XGBRegressor for prediction
        """
        model = XGBRegressor()
        if path :
            self.regressor.load_model("pydistinct/xgb_estimator.pkl")
        else :
            self.regressor.load_model(path)

    def predict(self, data, *args):
        result = est_transform(data)
        return self.regressor.predict(result)


def est_transform(sequence, features=False):
    """

    :param sequence: sample sequence of integers
    :type sequence: array of ints
    :param features:
    :type boolean: If True, requires array with feature and value tuples [(feature,value),]
    :returns: raw values for training XGBRegressor
    """
    # Check is fit had been called

    # Input validation
    # sequence = check_array(sequence, accept_sparse=False)

    n = len(sequence)
    d = len(set(sequence))
    n_bar = n / d

    attribute_counts = _get_attribute_counts(sequence)
    ac_values = list(attribute_counts.values())
    frequency_count = _get_frequency_dictionary(sequence)

    mu = sum((((i - n_bar) ** 2) / n_bar) for i in attribute_counts.values())
    chi_critical = chi2.isf(0.975, n - 1, loc=n_bar, scale=n_bar)  # set alpha is 0.975

    # wrapper function to try to get values if not return 0
    def try_wrap(func, **kwargs):
        try:
            return func(**kwargs)
        except:
            return d

    statistics = [("length of sequence", n),
                  ("distinct number of values", d),
                  ("length/distinct", n_bar),
                  ("chi_square comparision", 1 if mu > chi_critical else 0),
                  ("chi_square and critical ratio", mu / chi_critical),
                  ("attribute mean", np.mean(ac_values)),
                  ("attribute median", np.median(ac_values)),
                  ("attribute max", max(ac_values)),
                  ("attribute min", min(ac_values)),
                  ("attribute var", np.var(ac_values)),
                  ("attribute skew", skew(ac_values)),
                  ("attribute kurtosis", kurtosis(ac_values)),
                  ("attribute IRQ", (np.percentile(ac_values, 75) - np.percentile(ac_values, 25))),
                  ("attribute IRQ over median",
                   (np.percentile(ac_values, 75) - np.percentile(ac_values, 25)) / np.median(ac_values)),
                  ("attribute MAD over median", robust.mad(ac_values) / np.median(ac_values)),
                  ("attribute coefficient of variance", np.var(ac_values) / np.mean(ac_values))
                  ]
    if 1 in frequency_count:
        statistics.append(("f1", frequency_count[1]))
    else:
        statistics.append(("f1", 0))

    if 2 in frequency_count:
        statistics.append(("f2", frequency_count[2]))
    else:
        statistics.append(("f2", 0))

    estimators = [("chao_estimator", try_wrap(chao_estimator, sequence=sequence)),
                  ("chao_lee_estimator", try_wrap(chao_lee_estimator, sequence=sequence)),
                  ("jackknife_estimator", try_wrap(jackknife_estimator, sequence=sequence)),
                  ("sichel_estimator", try_wrap(sichel_estimator, sequence=sequence)),
                  ("bootstrap_estimator", try_wrap(bootstrap_estimator, sequence=sequence)),
                  ("method_of_moments_estimator", try_wrap(method_of_moments_estimator, sequence=sequence))
                  ]
    """
    for i in [2]:
        estimators.append(("shlossers_estimator_{}".format(str(i)),
                           try_wrap(shlossers_estimator, sequence=sequence, pop_estimator=lambda x: x * i)))
    """
    for n_pop in [1000, 10000, 100000, 1000000, 10000000]:
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
     #   estimators.append(("hybrid_estimator_{}".format(n_pop),
      #                     try_wrap(hybrid_estimator, sequence=sequence, pop_estimator=lambda x: x * n_pop)))

    results = statistics + estimators

    def clean_result(result):
        """
        clean result if nan, list or np.array - return float
        """
        key = result[0]
        value = result[1]
        if np.isnan(value):
            return (key, 0)
        if type(value) == list or isinstance(value, np.ndarray):
            return key, np.float(value)
        return result

    if features:
        array_of_results = [clean_result(result) for result in results]
    else:  # default : return only raw values
        array_of_results = [clean_result(result)[1] for result in results]

    return array_of_results
