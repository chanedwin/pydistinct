# Distinct Value Estimators

This package provides statistical estimators to predict a population's total number of distinct values D from a sample sequence. In essence : given a sample of n values with only d distinct values from a population N, predict the total number of distinct values D that exists in the population.

Use cases : 
* estimating the number of unique insects in a population from a field sample,
* estimating the number of unique items in a database from a sample 

This package is based on work from Haas et al, 1995 with estimators from Goodman 1949, Ozsoyoglu et al., 1991, Chao 1984, Chao and Lee 1992, Shlosser 1981, Sichel 1986a, 1986b and 1992,Bunge and Fitzpatrick 1993, Smith and Van Bell 1984, Sarndal,
Swensson, and Wretman 1992, Burnham and Overton 1979.
 
It provides a python implementation for the statistical estimators in Haas as well as an XGB ensemble estimator 
to predict the total number [In progress].

Use case : Given a sample of integers with d distinct values, predict the total number of distinct values D within a population 

## Installation

pip install pydistinct

## Usage
```
from pydistinct.stats_estimators import *
uniform = sampling.sample_gaussian(200,1000,500)
print(uniform)
>>> {"sample":[-252, -238, -109.. 302, 122], 'sample_distinct': 359, 'ground_truth': 552}
bootstrap_estimator(uniform["sample"])
>>> 463.7695215710723
horvitz_thompson_estimator(uniform["sample"])
>>> 519.6486453398398
method_of_moments_v3_estimator(uniform["sample"])
>>> 709.4574356684974
```

## Estimators available : 
* goodmans_estimator : 
* chao_estimator : 
* chao_lee_estimator : 
* jackknife_estimator : 
* sichel_estimator :
* bootstrap_estimator :
* method_of_moments_estimator :
* shlossers_estimator :
* horvitz_thompson_estimator :
* method_of_moments_estimator :
* method_of_moments_v2_estimator :
* method_of_moments_v3_estimator :
* smoothed_jackknife_estimator :
* hybrid_estimator : 


## Additional planned work

* Include automatic techniques to convert strings to integers

## References

Haas, P. J., Naughton, J. F., Seshadri, S., & Stokes, L. (1995, September). Sampling-based estimation of the number of distinct values of an attribute. In VLDB (Vol. 95, pp. 311-322).

Goodman, L. A. (1949). On the estimation of the number of classes in a population. The Annals of Mathematical Statistics, 20(4), 572-579.

Bunge, J., & Fitzpatrick, M. (1993). Estimating the number of species: a review. Journal of the American Statistical Association, 88(421), 364-373.

Burnham, K. P., & Overton, W. S. (1978). Estimation of the size of a closed population when capture probabilities vary among animals. Biometrika, 65(3), 625-633.

Chao, A. (1984). Nonparametric estimation of the number of classes in a population. Scandinavian Journal of statistics, 265-270.

Chao, A., & Lee, S. M. (1992). Estimating the number of classes via sample coverage. Journal of the American statistical Association, 87(417), 210-217.

Heltshe, J. F., & Forrester, N. E. (1983). Estimating species richness using the jackknife procedure. Biometrics, 1-11.

Ozsoyoglu, G., Du, K., Tjahjana, A., Hou, W. C., & Rowland, D. Y. (1991). On estimating COUNT, SUM, and AVERAGE relational algebra queries. In Database and Expert Systems Applications (pp. 406-412). Springer, Vienna.

Särndal, C. E., Swensson, B., & Wretman, J. (2003). Model assisted survey sampling. Springer Science & Business Media.

Shlosser, A. (1981). On estimation of the size of the dictionary of a long text on the basis of a sample. Engineering Cybernetics, 19(1), 97-102.

Sichel, H. S. (1986). Parameter estimation for a word frequency distribution based on occupancy theory. Communications in Statistics-Theory and Methods, 15(3), 935-949.

Sichel, H. S. (1986). Word frequency distributions and type-token characteristics. Math. Scientist, 11, 45-72.

Sichel, H. S. (1992). Anatomy of the generalized inverse Gaussian-Poisson distribution with special applications to bibliometric studies. Information Processing & Management, 28(1), 5-17.

Smith, E. P., & van Belle, G. (1984). Nonparametric estimation of species richness. Biometrics, 119-129.
