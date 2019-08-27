https://pydistinct.readthedocs.io/en/0.5/ 

# Pydistinct - Population Distinct Value Estimators

This python package provides statistical estimators to predict a population's cardinality (number of distinct values) from a sample sequence of that population, like estimating the unique elements in a database with only a few rows or the number of unique twitter users tweeting about a topic with only the latest thousand tweets.


Please send all bugs reports/issues/queries to chanedwin91@gmail.com for fastest response! 

## Installation

pip install pydistinct

## Requirements

numpy, statsmodels, scipy

## Usage

```python
from pydistinct.stats_estimators import *
sequence = [1,2,2,3]
horvitz_thompson_estimator(sequence)
>>> 3.9923808687325613

from pydistinct.sampling import sample_uniform
uniform = sample_uniform() # sample 500 values from a distribution of 1000 integers with uniform probability
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

# if you know the population size or the ratio of population size to sample size 
from pydistinct.sampling import sample_gaussian
gaussian = sample_gaussian(population_size=1000,sample_size=500) # gaussian distribution centered at 0

# if you know the population size or the ratio of population size to sample size 
smoothed_jackknife_estimator(uniform["sample"]) # provide population size  
>>> 766.6902930100636 #algorithm overpredicts

smoothed_jackknife_estimator(uniform["sample"],pop_estimator = lambda x : x * 2) # provide ratio of sample size to population 
>>>505.1381557576779
smoothed_jackknife_estimator(uniform["sample"],n_pop= 1000) # provide actual population size
>>> 505.1381557576779

# Currently, all the estimators only take in sequences of integers. 
# You will need to use a label encoder to convert strings to integers.
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
my_string = ['two', 'roads', 'diverged'... 'in', 'the', 'undergrowth'] # first paragraph of Frost's The Road Not Taken
sequence = le.fit_transform(my_string)
print(sequence)
>>> [24 17  7 11  0 28 27  1 18 10  6 15 22  5  1  3 16 23 13 10 19  1 14  8
 16  2  9  2 10  6 21 26 12  4 11 20 25]
smoothed_jackknife_estimator(sequence)
>>> 74.99998801158523    # ground truth : 95 unique words in the poem



```

## Estimators available 

Suggested Estimators (See Usage)
* bootstrap_estimator : More conservative technique, could also be useful as a lower bound. Useful if you want to avoid overpredicting
* smoothed_jackknife_estimator : More aggressive technique, could also be useful as a upper bound, Useful if you want to avoid underpredicting
* horvitz_thompson_estimator : tends to predict somewhat in the middle of bootstrap and smoothed jackknife

This package is based on work from [Haas et al, 1995](https://pdfs.semanticscholar.org/d26b/70479bc818ef7079732ba014e82368dbf66f.pdf) with estimators from Goodman (1949), Chao (1984), Chao and Lee (1992), Shlosser (1981), Sichel (1986a, 1986b and 1992), Smith and Van Bell (1984), Sarndal,
Swensson, and Wretman (1992),

* **goodmans_estimator** : Implementation of Goodman's estimator, unique unbiased estimator of D
* **chao_estimator** : Implementation of Chao's estimator, using counts of values that appear exactly once and twice
* **jackknife_estimator** : Jackknife scheme for estimating D 
* **chao_lee_estimator** : Implementation of Chao and Lee's estimator  using a natural estimator of coverage 
* **shlossers_estimator** : Implementation of Shlosser's Estimator using a Bernoulli Sampling scheme
* **sichel_estimator** : Implementation of Sichel’s Parametric Estimator which uses a zero-truncated generalized inverse Gaussian-Poisson to estimate D
* **method_of_moments_estimator** : Simple Method-of-Moments Estimator to estimate D 
* **bootstrap_estimator** : Implementation of a bootstrap estimator to estimate D 
* **horvitz_thompson_estimator** : Implementation of the Horvitz-Thompson Estimator to estimate D 
* **method_of_moments_v2_estimator** : Method-of-Moments Estimator with equal frequency assumption while still sampling from a finite relation
* **method_of_moments_v3_estimator** : Method-of-Moments Estimator without equal frequency assumption 
* **smoothed_jackknife_estimator** : Jackknife scheme for estimating D that accounts for true bias structures 
* **hybrid_estimator** : Hybrid Estimator that uses Shlosser's estimator when data is skewed and Smooth jackknife estimator when data is not. Skew is computed by using an approximate chi square test for uniformity

## Sample Use Cases

estimating the number of 
   * unique elements in an infinite stream 
   * unique insects in a population from a field sample
   * unique words in a document given a sentence or a paragraph
   * estimating the number of unique items in a database from a few sample rows
   
## Complexities
### All values are unique
Where all values seen are unique (d unique values in sequence of length d), no statistical method works and the methods fall back to a special case of [birthday problem](https://en.wikipedia.org/wiki/Birthday_problem) with no collisions. In this problem, we try different values of the distinct values in the population (D), and estimate the probability that we draw d unique values from it with no collision. Intuitively, if our sample contains 10 unique values, then D is more likely to be 100 than 10. If we set a posterior probability (default 0.1), we can then compute the smallest value for D where the probability is greater than 0.1. You can tweak the probability of the birthday solution to get the lower bound (around 0.1) or an upper bound estimate (something like 0.9) of D.

### Knowledge of population size (N) 

In most real world problems, the population size N will not be known - all that is available is the sample sequence. Most of estimators would be improved if the population size N is given to it, but if it isn't the estimators would just assume a very large N and attempt to estimate D anyway. However, in cases where the population size is known, the estimators that rely on population size will take a value (n_pop = N) or a function (pop_estimator that takes in a function of sample size and returns population size i.e lambda x : x * 10) and use that at prediction time.

### Comparison with hyperloglog

Comparison of these estimators with [hyperloglog](https://pypi.org/project/hyperloglog/) - if you have the entire dataset and you want to compute its cardinality, use hyperloglog. If you only have a sample of the dataset but want to estimate the entire dataset's cardinality, these estimators would be more appropriate.

## Additional planned work

* Include ensemble learning to improve on estimator predictions
* Include newer work on streaming algorithms


## References

Haas, P. J., Naughton, J. F., Seshadri, S., & Stokes, L. (1995, September). Sampling based estimation of the number of distinct values of an attribute. In VLDB (Vol. 95, pp. 311-322).

Bunge, J., & Fitzpatrick, M. (1993). Estimating the number of species: a review. Journal of the American Statistical Association, 88(421), 364-373.

Burnham, K. P., & Overton, W. S. (1978). Estimation of the size of a closed population when capture probabilities vary among animals. Biometrika, 65(3), 625-633.

Chao, A. (1984). Nonparametric estimation of the number of classes in a population. Scandinavian Journal of statistics, 265-270.

Chao, A., & Lee, S. M. (1992). Estimating the number of classes via sample coverage. Journal of the American statistical Association, 87(417), 210-217.

Goodman, L. A. (1949). On the estimation of the number of classes in a population. The Annals of Mathematical Statistics, 20(4), 572-579.

Heltshe, J. F., & Forrester, N. E. (1983). Estimating species richness using the jackknife procedure. Biometrics, 1-11.

Ozsoyoglu, G., Du, K., Tjahjana, A., Hou, W. C., & Rowland, D. Y. (1991). On estimating COUNT, SUM, and AVERAGE relational algebra queries. In Database and Expert Systems Applications (pp. 406-412). Springer, Vienna.

Särndal, C. E., Swensson, B., & Wretman, J. (2003). Model assisted survey sampling. Springer Science & Business Media.

Shlosser, A. (1981). On estimation of the size of the dictionary of a long text on the basis of a sample. Engineering Cybernetics, 19(1), 97-102.

Sichel, H. S. (1986). Parameter estimation for a word frequency distribution based on occupancy theory. Communications in Statistics-Theory and Methods, 15(3), 935-949.

Sichel, H. S. (1986). Word frequency distributions and type-token characteristics. Math. Scientist, 11, 45-72.

Sichel, H. S. (1992). Anatomy of the generalized inverse Gaussian-Poisson distribution with special applications to bibliometric studies. Information Processing & Management, 28(1), 5-17.

Smith, E. P., & van Belle, G. (1984). Nonparametric estimation of species richness. Biometrics, 119-129.

## Special Thanks

[Keng Hwee](https://github.com/kenghweeng)

### See Documentation at https://pydistinct.readthedocs.io/en/0.5/ 

Written in python
