import unittest

from pydistinct.ml_estimator import *
from pydistinct.sampling import *


class TestEstimatorMethods(unittest.TestCase):

    def setUp(self):
        self.unique_sequence = np.array([1, 2, 3, 4])
        self.test_sequence = np.array([1, 2, 3, 3, 4, 4, 5, 5, 5])
        self.uniform_sequence = sample_uniform(seed=42)["sample"]
        self.gaussian_sequence = sample_gaussian(seed=42)["sample"]
        self.zipf_sequence = sample_zipf(seed=42)["sample"]

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


class TestMLEstimatorMethods(unittest.TestCase):
    def test_transformer(self):
        transform_1_features = est_transform([1, 2, 3, 4, 5, 6], features=True)
        transform_1_values = est_transform([1, 2, 3, 4, 5, 6], features=False)

        self.assertEqual(transform_1_features, [('length of sequence', 6),
                                                ('distinct number of values', 6),
                                                ('length/distinct', 1.0),
                                                ('chi_square comparision', 0),
                                                ('chi_square and critical ratio', 0.0),
                                                ('attribute mean', 1.0),
                                                ('attribute median', 1.0),
                                                ('attribute max', 1),
                                                ('attribute min', 1),
                                                ('attribute var', 0.0),
                                                ('attribute skew', 0.0),
                                                ('attribute kurtosis', -3.0),
                                                ('attribute IRQ', 0.0),
                                                ('attribute IRQ over median', 0.0),
                                                ('attribute MAD over median', 0.0),
                                                ('attribute coefficient of variance', 0.0),
                                                ('f1', 6),
                                                ('f2', 0),
                                                ('chao_estimator', 12),
                                                ('chao_lee_estimator', 12),
                                                ('jackknife_estimator', 12),
                                                ('sichel_estimator', 12),
                                                ('bootstrap_estimator', 12),
                                                ('method_of_moments_estimator', 12),
                                                ('shlossers_estimator_2', 12),
                                                ('shlossers_estimator_3', 12),
                                                ('shlossers_estimator_5', 12),
                                                ('shlossers_estimator_10', 12),
                                                ('shlossers_estimator_20', 12),
                                                ('horvitz_thompson_estimator_1000', 12),
                                                ('method_of_moments_v2_estimator_1000', 12),
                                                ('method_of_moments_v3_estimator_1000', 12),
                                                ('smoothed_jackknife_estimator_1000', 12),
                                                ('hybrid_estimator_1000', 12),
                                                ('horvitz_thompson_estimator_10000', 12),
                                                ('method_of_moments_v2_estimator_10000', 12),
                                                ('method_of_moments_v3_estimator_10000', 12),
                                                ('smoothed_jackknife_estimator_10000', 12),
                                                ('hybrid_estimator_10000', 12),
                                                ('horvitz_thompson_estimator_100000', 12),
                                                ('method_of_moments_v2_estimator_100000', 12),
                                                ('method_of_moments_v3_estimator_100000', 12),
                                                ('smoothed_jackknife_estimator_100000', 12),
                                                ('hybrid_estimator_100000', 12),
                                                ('horvitz_thompson_estimator_1000000', 12),
                                                ('method_of_moments_v2_estimator_1000000', 12),
                                                ('method_of_moments_v3_estimator_1000000', 12),
                                                ('smoothed_jackknife_estimator_1000000', 12),
                                                ('hybrid_estimator_1000000', 12),
                                                ('horvitz_thompson_estimator_10000000', 12),
                                                ('method_of_moments_v2_estimator_10000000', 12),
                                                ('method_of_moments_v3_estimator_10000000', 12),
                                                ('smoothed_jackknife_estimator_10000000', 12),
                                                ('hybrid_estimator_10000000', 12)])

        self.assertEqual(transform_1_values, [6,
                                              6,
                                              1.0,
                                              0,
                                              0.0,
                                              1.0,
                                              1.0,
                                              1,
                                              1,
                                              0.0,
                                              0.0,
                                              -3.0,
                                              0.0,
                                              0.0,
                                              0.0,
                                              0.0,
                                              6,
                                              0,
                                              12,
                                              12,
                                              12,
                                              12,
                                              12,
                                              12,
                                              12,
                                              12,
                                              12,
                                              12,
                                              12,
                                              12,
                                              12,
                                              12,
                                              12,
                                              12,
                                              12,
                                              12,
                                              12,
                                              12,
                                              12,
                                              12,
                                              12,
                                              12,
                                              12,
                                              12,
                                              12,
                                              12,
                                              12,
                                              12,
                                              12,
                                              12,
                                              12,
                                              12,
                                              12,
                                              12])

        transform_2_features = est_transform(sample_zipf(seed=42)["sample"], features=True)
        transform_2_values = est_transform(sample_zipf(seed=42)["sample"], features=False)

        self.assertEqual(transform_2_features, [('length of sequence', 500),
                                                ('distinct number of values', 157),
                                                ('length/distinct', 3.1847133757961785),
                                                ('chi_square comparision', 1),
                                                ('chi_square and critical ratio', 6.159060609132323),
                                                ('attribute mean', 3.1847133757961785),
                                                ('attribute median', 1.0),
                                                ('attribute max', 158),
                                                ('attribute min', 1),
                                                ('attribute var', 175.06779179682738),
                                                ('attribute skew', 10.471992362743215),
                                                ('attribute kurtosis', 117.1545559111356),
                                                ('attribute IRQ', 0.0),
                                                ('attribute IRQ over median', 0.0),
                                                ('attribute MAD over median', 0.0),
                                                ('attribute coefficient of variance', 54.97128662420379),
                                                ('f1', 119),
                                                ('f2', 12),
                                                ('chao_estimator', 747.0416666666666),
                                                ('chao_lee_estimator', 223.20625487737598),
                                                ('jackknife_estimator', 275.7619999999998),
                                                ('sichel_estimator', 9504288.160310695),
                                                ('bootstrap_estimator', 202.93751628219297),
                                                ('method_of_moments_estimator', 164.96233450766724),
                                                ('shlossers_estimator_2', 264.63242074716965),
                                                ('shlossers_estimator_3', 357.7987112874076),
                                                ('shlossers_estimator_5', 525.5291971698468),
                                                ('shlossers_estimator_10', 898.1151513409612),
                                                ('shlossers_estimator_20', 1549.22746410288),
                                                ('horvitz_thompson_estimator_1000', 228.57048851340548),
                                                ('method_of_moments_v2_estimator_1000', 164.85628552487307),
                                                ('method_of_moments_v3_estimator_1000', 164.16769822819973),
                                                ('smoothed_jackknife_estimator_1000', 205.66159907264472),
                                                ('hybrid_estimator_1000', 39506.967562363774),
                                                ('horvitz_thompson_estimator_10000', 228.62256595549053),
                                                ('method_of_moments_v2_estimator_10000', 164.86927024121238),
                                                ('method_of_moments_v3_estimator_10000', 164.8008246394902),
                                                ('smoothed_jackknife_estimator_10000', 205.71800998317588),
                                                ('hybrid_estimator_10000', 375830.6254166934),
                                                ('horvitz_thompson_estimator_100000', 228.62778217067145),
                                                ('method_of_moments_v2_estimator_100000', 164.87148373953977),
                                                ('method_of_moments_v3_estimator_100000', 164.8645518101353),
                                                ('smoothed_jackknife_estimator_100000', 205.72376831461546),
                                                ('hybrid_estimator_100000', 3738773.386692761),
                                                ('horvitz_thompson_estimator_1000000', 228.62834400756216),
                                                ('method_of_moments_v2_estimator_1000000', 164.87165338534325),
                                                ('method_of_moments_v3_estimator_1000000', 164.87096393364985),
                                                ('smoothed_jackknife_estimator_1000000', 205.72434532351863),
                                                ('hybrid_estimator_1000000', 37368173.66067749),
                                                ('horvitz_thompson_estimator_10000000', 228.627688009936),
                                                ('method_of_moments_v2_estimator_10000000', 157),
                                                ('method_of_moments_v3_estimator_10000000', 164.87169821498233),
                                                ('smoothed_jackknife_estimator_10000000', 205.7244030361701),
                                                ('hybrid_estimator_10000000', 373662173.68805337)])
        self.assertEqual(transform_2_values, [500,
                                              157,
                                              3.1847133757961785,
                                              1,
                                              6.159060609132323,
                                              3.1847133757961785,
                                              1.0,
                                              158,
                                              1,
                                              175.06779179682738,
                                              10.471992362743215,
                                              117.1545559111356,
                                              0.0,
                                              0.0,
                                              0.0,
                                              54.97128662420379,
                                              119,
                                              12,
                                              747.0416666666666,
                                              223.20625487737598,
                                              275.7619999999998,
                                              9504288.160310695,
                                              202.93751628219297,
                                              164.96233450766724,
                                              264.63242074716965,
                                              357.7987112874076,
                                              525.5291971698468,
                                              898.1151513409612,
                                              1549.22746410288,
                                              228.57048851340548,
                                              164.85628552487307,
                                              164.16769822819973,
                                              205.66159907264472,
                                              39506.967562363774,
                                              228.62256595549053,
                                              164.86927024121238,
                                              164.8008246394902,
                                              205.71800998317588,
                                              375830.6254166934,
                                              228.62778217067145,
                                              164.87148373953977,
                                              164.8645518101353,
                                              205.72376831461546,
                                              3738773.386692761,
                                              228.62834400756216,
                                              164.87165338534325,
                                              164.87096393364985,
                                              205.72434532351863,
                                              37368173.66067749,
                                              228.627688009936,
                                              157,
                                              164.87169821498233,
                                              205.7244030361701,
                                              373662173.68805337])

    def test_estimator(self):
        model = load_pretrained_model()
        test_model = XGBRegressor()
        self.assertEqual(type(model), type(test_model))


if __name__ == '__main__':
    unittest.main()
