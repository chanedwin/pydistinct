import unittest

import numpy as np

from pydistinct.bootstrap import bootstrap, BootstrapResults
from pydistinct.ensemble_estimators import *
from pydistinct.sampling import *
from pydistinct.utils import _compute_birthday_problem_probability, _get_attribute_counts, _get_frequency_dictionary


class TestEstimatorMethods(unittest.TestCase):

    def setUp(self):
        self.unique_sequence = np.array([1, 2, 3, 4])
        self.test_sequence = np.array([1, 2, 3, 3, 4, 4, 5, 5, 5])
        self.uniform_sequence = sample_uniform(seed=42)["sample"]
        self.gaussian_sequence = sample_gaussian(seed=42)["sample"]
        self.zipf_sequence = sample_zipf(seed=42)["sample"]
        self.small_zipf_sequence = sample_zipf(seed=42, sample_size=20)["sample"]

    def test_birthday_probability(self):
        self.assertEqual(_compute_birthday_problem_probability(self.unique_sequence), 8)

    def test_attr_count(self):
        self.assertEqual(_get_attribute_counts(self.test_sequence), {1: 1, 2: 1, 3: 2, 4: 2, 5: 3})
        self.assertEqual(_get_attribute_counts(self.small_zipf_sequence),
                         {1: 7, 2: 2, 3: 4, 6: 1, 19: 1, 22: 1, 23: 1, 46: 1, 80: 1, 385: 1}
                         )

    def test_freq_count(self):
        self.assertEqual(_get_frequency_dictionary(self.test_sequence), {1: 2, 2: 2, 3: 1})
        self.assertEqual(_get_frequency_dictionary(self.small_zipf_sequence), {1: 7, 2: 1, 4: 1, 7: 1}
                         )

    def test_goodmans(self):
        self.assertEqual(goodmans_estimator(self.unique_sequence), 8)
        self.assertEqual(goodmans_estimator(self.test_sequence), 6.464285714285714)
        self.assertEqual(goodmans_estimator(self.uniform_sequence), 571.8774533444981)
        self.assertEqual(goodmans_estimator(self.gaussian_sequence), 555.9082956099103)
        self.assertEqual(goodmans_estimator(self.zipf_sequence), -8.269478047057136e+21)

    def test_chao(self):
        self.assertEqual(chao_estimator(self.unique_sequence), 8)
        self.assertEqual(chao_estimator(self.test_sequence), 6.0)
        self.assertEqual(chao_estimator(self.uniform_sequence), 802.7202380952381)
        self.assertEqual(chao_estimator(self.gaussian_sequence), 761.05625)
        self.assertEqual(chao_estimator(self.zipf_sequence), 747.0416666666666)

    def test_jackknife(self):
        self.assertEqual(jackknife_estimator(self.unique_sequence), 8)
        self.assertEqual(jackknife_estimator(self.test_sequence), 6.777777777777779)
        self.assertEqual(jackknife_estimator(self.uniform_sequence), 640.4620000000054)
        self.assertEqual(jackknife_estimator(self.gaussian_sequence), 613.4939999999859)
        self.assertEqual(jackknife_estimator(self.zipf_sequence), 275.7619999999998)

    def test_chao_lee(self):
        self.assertEqual(chao_lee_estimator(self.unique_sequence), 8.0)
        self.assertEqual(chao_lee_estimator(self.test_sequence), 6.517460317460317)
        self.assertEqual(chao_lee_estimator(self.uniform_sequence), 805.5440556719267)
        self.assertEqual(chao_lee_estimator(self.gaussian_sequence), 731.0836456985209)
        self.assertEqual(chao_lee_estimator(self.zipf_sequence), 223.20625487737598)

    def test_shlossers(self):
        self.assertEqual(shlossers_estimator(self.unique_sequence), 8.0)
        self.assertEqual(shlossers_estimator(self.test_sequence), 6.368421052631579)
        self.assertEqual(shlossers_estimator(self.uniform_sequence), 603.6873932353947)
        self.assertEqual(shlossers_estimator(self.gaussian_sequence), 575.5934182590233)
        self.assertEqual(shlossers_estimator(self.zipf_sequence), 264.63242074716965)

    def test_sichel(self):
        self.assertEqual(sichel_estimator(self.unique_sequence), 8)
        self.assertEqual(sichel_estimator(self.test_sequence), 5)
        self.assertEqual(sichel_estimator(self.uniform_sequence), 836.7891724965193)
        self.assertEqual(sichel_estimator(self.gaussian_sequence), 789.713450987392)
        self.assertEqual(sichel_estimator(self.zipf_sequence), 9504288.160310695)

    def test_bootstrap(self):
        self.assertEqual(bootstrap_estimator(self.unique_sequence), 8)
        self.assertEqual(bootstrap_estimator(self.test_sequence), 5.927210553389188)
        self.assertEqual(bootstrap_estimator(self.uniform_sequence), 482.9724642531561)
        self.assertEqual(bootstrap_estimator(self.gaussian_sequence), 466.0513843659596)
        self.assertEqual(bootstrap_estimator(self.zipf_sequence), 202.93751628219297)

    def test_horvitz_thompson(self):
        self.assertEqual(horvitz_thompson_estimator(self.unique_sequence), 8)
        self.assertEqual(horvitz_thompson_estimator(self.test_sequence), 6.319408087820948)
        self.assertEqual(horvitz_thompson_estimator(self.uniform_sequence), 542.2174207208585)
        self.assertEqual(horvitz_thompson_estimator(self.gaussian_sequence), 521.8215783111308)
        self.assertEqual(horvitz_thompson_estimator(self.zipf_sequence), 228.627688009936)

    def test_method_of_moments_one(self):
        self.assertEqual(method_of_moments_estimator(self.unique_sequence), 8)
        self.assertEqual(method_of_moments_estimator(self.test_sequence), 6.8265909343805165)
        self.assertEqual(method_of_moments_estimator(self.uniform_sequence), 801.6089111589724)
        self.assertEqual(method_of_moments_estimator(self.gaussian_sequence), 723.4787989985088)
        self.assertEqual(method_of_moments_estimator(self.zipf_sequence), 164.96233450766724)

    def test_method_of_moments_two(self):
        self.assertEqual(method_of_moments_v2_estimator(self.unique_sequence), 8)
        self.assertEqual(method_of_moments_v2_estimator(self.test_sequence), 6.3694941686386155)
        self.assertEqual(method_of_moments_v2_estimator(self.uniform_sequence), 372)
        self.assertEqual(method_of_moments_v2_estimator(self.gaussian_sequence), 722.3400395056569)
        self.assertEqual(method_of_moments_v2_estimator(self.zipf_sequence), 164.87165338534325)

    def test_method_of_moments_three(self):
        self.assertEqual(method_of_moments_v3_estimator(self.unique_sequence), 8)
        self.assertEqual(method_of_moments_v3_estimator(self.test_sequence), 6.369496180824973)
        self.assertEqual(method_of_moments_v3_estimator(self.uniform_sequence), 502.91263927254766)
        self.assertEqual(method_of_moments_v3_estimator(self.gaussian_sequence), 722.3427899336142)
        self.assertEqual(method_of_moments_v3_estimator(self.zipf_sequence), 164.87169821498233)

    def test_smooth_jackknife(self):
        self.assertEqual(smoothed_jackknife_estimator(self.unique_sequence), 8)
        self.assertEqual(smoothed_jackknife_estimator(self.test_sequence), 6.142856986848082)
        self.assertEqual(smoothed_jackknife_estimator(self.uniform_sequence), 804.0302095880729)
        self.assertEqual(smoothed_jackknife_estimator(self.gaussian_sequence), 729.7448646736057)
        self.assertEqual(smoothed_jackknife_estimator(self.zipf_sequence), 205.7244030361701)

    def test_hybrid(self):
        self.assertEqual(hybrid_estimator(self.unique_sequence), 8)
        self.assertEqual(hybrid_estimator(self.test_sequence), 6.142856986848082)
        self.assertEqual(hybrid_estimator(self.uniform_sequence), 804.0302095880729)
        self.assertEqual(hybrid_estimator(self.gaussian_sequence), 729.7448646736057)
        self.assertEqual(hybrid_estimator(self.zipf_sequence), 373662173.68805337)

    def test_median(self):
        self.assertEqual(median_estimator(self.unique_sequence), 8)
        self.assertEqual(median_estimator(self.test_sequence), 6.366099596631006)
        self.assertGreaterEqual(median_estimator(self.uniform_sequence), 796.140916441000)  # last dp is diff on diff os
        self.assertLessEqual(median_estimator(self.uniform_sequence), 796.140916441010)
        self.assertEqual(median_estimator(self.gaussian_sequence), 722.3281454754505)
        self.assertEqual(median_estimator(self.zipf_sequence), 205.66159907264472)

    def test_cached_funcion(self):  # critical to test cache function on all functions

        # build cache
        n = len(self.test_sequence)
        d = len(set(self.test_sequence))
        attribute_counts = _get_attribute_counts(self.test_sequence)
        frequency_dictionary = _get_frequency_dictionary(self.test_sequence)
        cached_data = {"n": n, "d": d, "attr": attribute_counts, "freq": frequency_dictionary}

        # test cache
        self.assertEqual(goodmans_estimator(self.test_sequence, cache=cached_data), 6.464285714285714)
        self.assertEqual(chao_estimator(self.test_sequence, cache=cached_data), 6.0)
        self.assertEqual(jackknife_estimator(self.test_sequence, cache=cached_data), 6.777777777777779)
        self.assertEqual(chao_lee_estimator(self.test_sequence, cache=cached_data), 6.517460317460317)
        self.assertEqual(shlossers_estimator(self.test_sequence, cache=cached_data), 6.368421052631579)
        self.assertEqual(sichel_estimator(self.test_sequence, cache=cached_data), 5)
        self.assertEqual(bootstrap_estimator(self.test_sequence, cache=cached_data), 5.927210553389188)
        self.assertEqual(horvitz_thompson_estimator(self.test_sequence, cache=cached_data), 6.319408087820948)
        self.assertEqual(method_of_moments_estimator(self.test_sequence, cache=cached_data), 6.8265909343805165)
        self.assertEqual(method_of_moments_v2_estimator(self.test_sequence, cache=cached_data), 6.3694941686386155)
        self.assertEqual(method_of_moments_v3_estimator(self.test_sequence, cache=cached_data), 6.369496180824973)
        self.assertEqual(smoothed_jackknife_estimator(self.test_sequence, cache=cached_data), 6.142856986848082)
        self.assertEqual(hybrid_estimator(self.test_sequence, cache=cached_data), 6.142856986848082)

    def test_bootstrap(self):
        self.assertEqual(bootstrap(self.unique_sequence, stat_func=median_estimator),
                         BootstrapResults(3.1999999999999993, 8, 12.6))
        self.assertEqual(bootstrap(self.test_sequence, stat_func=median_estimator),
                         BootstrapResults(3.237425335862243, 6.366099596631006 , 8.374788818911835))
        self.assertEqual(bootstrap(self.gaussian_sequence, stat_func=median_estimator),
                         BootstrapResults(668.1224408865861, 722.3281454754505 , 784.4087329433855))
        self.assertEqual(bootstrap(self.zipf_sequence, stat_func=median_estimator),
                         BootstrapResults(183.34641124915314, 205.66159907264472 , 225.47196616261735))

if __name__ == '__main__':
    unittest.main()
