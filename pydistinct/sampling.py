import numpy as np


def sample_uniform(n_distinct_integers=1000, sample_size=500, seed=None):
    """
    generate a sample sequence of integers from a uniform distribution of n distinct integers

    :param n_distinct_integers: N number of distinct integers to generate sample from
    :type n_distinct_integers: int
    :param sample_size: sample size
    :type sample_size: int
    :param seed: random seed for numpy rng generator
    :type seed: int
    :returns : dictionary with sampled sequence[sample],
                               actual number of distinct values[ground_truth],
                               sample number of distinct values[sample_distinct]
    """
    np.random.seed(seed)
    sample = np.random.randint(1, n_distinct_integers, sample_size)

    return {"sample": sample, "sample_distinct": len(set(sample)), "ground_truth": n_distinct_integers}


def sample_gaussian(cov=200.0, population_size=1000, sample_size=500, seed=None):
    """
    generate a ground truth population of distinct integers from a gaussian distribution (rounded to nearest int),
    then draw a sample sequence of integers without replacement from the population. Integers created could be negative

    :param cov: covariance of gaussian distribution
    :type cov: float
    :param population_size: ground truth number of distinct integers in population, with different probabilities
    :type population_size: int
    :param sample_size: sample size of sequence observed
    :type sample_size: int
    :param seed: random seed for numpy rng generator
    :type seed: int
    :returns : dictionary with sampled sequence[sample],
                               actual number of distinct values[ground_truth],
                               sample number of distinct values[sample_distinct]
    """

    np.random.seed(seed)
    population = [int(i) for i in np.random.normal(0, cov, population_size)]  # set mean as 0, doesn't really matter

    sample = population[:sample_size]

    return {"sample": sample, "sample_distinct": len(set(sample)), "ground_truth": len(set(population))}


def sample_zipf(alpha=1.3, population_size=1000, sample_size=500, seed=None):
    """
    generate a population of distinct integers from a zipf distribution (ground truth) as characterised by zipf(alpha,n)
    then draw a sample a sequence of integers from the population.

    :param alpha: alpha parameter of zipf distribution
    :type alpha: float
    :param population_size: the ground truth population size
    :type population_size: int
    :param sample_size: sample size of sequence observed
    :type sample_size: int
    :param seed: random seed for numpy rng generator
    :type seed: int
    :returns : dictionary with sampled sequence[sample],
                               actual number of distinct values[ground_truth],
                               sample number of distinct values[sample_distinct]
    """
    np.random.seed(seed)
    population = [int(i) for i in np.random.zipf(alpha, size=population_size)]

    sample = population[:sample_size]

    return {"sample": sample, "sample_distinct": len(set(sample)), "ground_truth": len(set(population))}
