# Pydistinct - Estimators for population distinct values from samples

This package provides statistical estimators to predict a population's total number of distinct values from a sample sequence - given a sample of n values with only d distinct values from a population N, predict the total number of distinct values D that exists in the population. 

Sample use cases :
* estimating the number of unique insects in a population from a field sample
* estimating the number of unique words in a document given a sentence or a paragraph
* estimating the number of unique items in a database from a few sample rows

This package is based on work from Haas et al, 1995 with estimators from Goodman 1949, Ozsoyoglu et al., 1991, Chao 1984, Chao and Lee 1992, Shlosser 1981, Sichel 1986a, 1986b and 1992,Bunge and Fitzpatrick 1993, Smith and Van Bell 1984, Sarndal,
Swensson, and Wretman 1992, Burnham and Overton 1979.
 
It provides a python implementation for the statistical estimators in Haas as well as an XGB ensemble estimator to predict the total number [In progress].


## Requirements

numpy, statsmodels, scipy, xgboost (Experimental)

## Installation

pip install pydistinct

## Usage

Currently, all the estimators only take in sequences of inte
```
from pydistinct.stats_estimators import *
sequence = [1,2,2,3]
horvitz_thompson_estimator(sequence)
>>> 3.9923808687325613

from pydistinct.sampling import sample_uniform
uniform = sample_uniform(n_distinct_integers=1000, sample_size=500) # sample 500 values from a uniform distribution of 1000 integers
print(uniform)
>>> {'ground_truth': 1000, # population distinct values
 'sample': array([ 50, 883, 190,... 797, 453, 867]), # 500 sampled values 
 'sample_distinct': 396} # only 396 distinct values in sample
 
bootstrap_estimator(uniform["sample"])
>>> 520.6409023638918 
horvitz_thompson_estimator(uniform["sample"])
>>> 588.8990980951648
smoothed_jackknife_estimator(uniform["sample"])
>>> 1057.1495560288624
```

## Estimators available (Haas et al, 1995) : 
* goodmans_estimator : Implementation of Goodman's estimator (Goodman 1949), unique unbiased estimator of D
* chao_estimator : Implementation of Chao's estimator (Chao 1984), using counts of values that appear exactly once and twice
* jackknife_estimator : Jackknife scheme for estimating D (Ozsoyoglu et al., 1991)
* chao_lee_estimator : Implementation of Chao and Lee's estimator (Chao and Lee, 1984) using a natural estimator of coverage 
* shlossers_estimator : Implementation of Shlosser's Estimator (Shlosser 1981) using a Bernoulli Sampling scheme
* sichel_estimator : Implementation of Sichel’s Parametric Estimator (Sichel 1986a, 1986b and 1992) which uses a zero-truncated generalized inverse Gaussian-Poisson to estimate D
* method_of_moments_estimator : Simple Method-of-Moments Estimator to estimate D (Haas et al, 1995)
* bootstrap_estimator : Implementation of a bootstrap estimator to estimate D (Smith and Van Bell 1984; Haas et al, 1995)
* horvitz_thompson_estimator : Implementation of the Horvitz-Thompson Estimator to estimate D (Sarndal,
Swensson, and Wretman 1992; Haas et al, 1995)
* method_of_moments_v2_estimator : Method-of-Moments Estimator with equal frequency assumption while still sampling from a finite relation (Haas et al, 1995)
* method_of_moments_v3_estimator : Method-of-Moments Estimator without equal frequency assumption (Haas et al, 1995)
* smoothed_jackknife_estimator : Jackknife scheme for estimating D that accounts for true bias structures (Haas et al, 1995)
* hybrid_estimator : Hybrid Estimator that uses Shlosser's estimator when data is skewed and Smooth jackknife estimator when data is not. Skew is computed by using an approximate chi square test for uniformity


## Complexities
Edge Case : Where all values seen are unique (d unique values in sequence of length d), no statistical method works and the methods fall back to a special case of [birthday problem](https://en.wikipedia.org/wiki/Birthday_problem) with no collisions. In this problem, we try different values of the distinct values in the population (D), and estimate the probability that we draw d unique values from it with no collision. Intuitively, if our sample contains 10 unique values, then D is more likely to be 100 than 10. If we set a posterior probability (default 0.1), we can then compute the smallest value for D where the probability is greater than 0.1. You can tweak the probability of the birthday solution to get the lower bound (around 0.1) or an upper bound estimate (something like 0.9) of D.

Computation of N (population size):

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

## Special Thanks

Ng Keng Hwee (
