# Distinct Value Estimators

This package is based on work from Haas et al, 1995.

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