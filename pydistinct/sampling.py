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
    generate a population of distinct integers from a zipf distribution (ground truth) as characterised by zipf(alpha,n),
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


def full_test(func, sequence, ground_truth, verbose=False, n_trials=100, params={}):
    """
    run estimator with a sequence for n trials and compares results with ground truth. Catches estimator errors if invalid

    :param func: estimator to be tested
    :type func: estimator function, takes in a sequence, returns a number (estimated distinct count)
    :param sequence: sequence of values observed
    :type sequence: array of ints
    :param ground_truth: actual number of distinct relations in population
    :type ground_truth: int
    :param verbose: Flag for verbosity of function
    :type verbose: boolean
    :param n_trials: number of trials to compute estimator, particularly important if random initialisation involved
    :type n_trials: int
    :param params: extra parameters to the estimator function
    :type params: dictionary
    :return: Prints results of tests
    :rtype: None
    """

    absolute_errors, percentage_errors, invalid_results_set, estimates = [], [], [], []
    invalid_count = 0
    for i in range(n_trials):  # n number of trials trials
        try:
            estimate = func(sequence, **params)
            estimates.append(estimate)
            absolute_errors.append(abs(estimate - ground_truth))
            percentage_errors.append(abs(estimate - ground_truth) / ground_truth)
        except Exception as e:
            if verbose:
                print("exception found :", e)
            invalid_count += 1
            invalid_results_set.append(e)
    print("average guess :", np.mean(estimates))
    print("overall absolute errors :", np.mean(np.nan_to_num(absolute_errors)))
    print("overall percentage errors :", np.mean(np.nan_to_num(percentage_errors)))
    print("invalid results observed :", invalid_count)
    print("errors are :", invalid_results_set)
