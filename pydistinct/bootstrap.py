# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# Original code from Facebook and bootstrapped library, modifications by Edwin Chan
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
'''Functions that allow one to create bootstrapped confidence intervals'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import multiprocessing as _multiprocessing

import numpy as _np
import scipy.sparse as _sparse


class BootstrapResults(object):
    def __init__(self, lower_bound, value, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.value = value
        if self.lower_bound > self.upper_bound:
            self.lower_bound, self.upper_bound = self.upper_bound, self.lower_bound

    def __str__(self):
        return '{1}    ({0}, {2})'.format(self.lower_bound, self.value,
                                          self.upper_bound)

    def __repr__(self):
        return self.__str__()

    def _apply(self, other, func):
        return BootstrapResults(func(self.lower_bound, other),
                                func(self.value, other),
                                func(self.upper_bound, other))

    def __add__(self, other):
        return self._apply(float(other), lambda x, other: other + x)

    def __radd__(self, other):
        return self._apply(float(other), lambda x, other: other + x)

    def __sub__(self, other):
        return self._apply(float(other), lambda x, other: x - other)

    def __rsub__(self, other):
        return self._apply(float(other), lambda x, other: other - x)

    def __mul__(self, other):
        return self._apply(float(other), lambda x, other: x * other)

    def __rmul__(self, other):
        return self._apply(float(other), lambda x, other: x * other)

    def __eq__(self, other):
        if type(other) != BootstrapResults:
            return 0
        return 1 if (
                self.lower_bound == other.lower_bound and
                self.upper_bound == other.upper_bound and
                self.value == other.value) \
            else 0

    def error_width(self):
        '''Returns: upper_bound - lower_bound'''
        return self.upper_bound - self.lower_bound

    def error_fraction(self):
        '''Returns the error_width / value'''
        if self.value == 0:
            return _np.inf
        else:
            return self.error_width() / self.value

    def is_significant(self):
        return _np.sign(self.upper_bound) == _np.sign(self.lower_bound)

    def get_result(self):
        '''Returns:
            -1 if statistically significantly negative
            +1 if statistically significantly positive
            0 otherwise
        '''
        return int(self.is_significant()) * _np.sign(self.value)


def _get_confidence_interval(bootstrap_dist, stat_val, alpha, is_pivotal):
    '''Get the bootstrap confidence interval for a given distribution.
    Args:
        bootstrap_distribution: numpy array of bootstrap results from
            bootstrap_distribution() or bootstrap_ab_distribution()
        stat_val: The overall statistic that this method is attempting to
            calculate error bars for.
        alpha: The alpha value for the confidence intervals.
        is_pivotal: if true, use the pivotal method. if false, use the
            percentile method.
    '''
    val = _np.percentile(bootstrap_dist, 50)
    bootstrap_dist = [i * (stat_val / val) for i in bootstrap_dist]  # necessary for this analysis
    if is_pivotal:
        low = 2 * stat_val - _np.percentile(bootstrap_dist, 100 * (1 - alpha / 2.))
        val = stat_val
        high = 2 * stat_val - _np.percentile(bootstrap_dist, 100 * (alpha / 2.))
    else:
        low = _np.percentile(bootstrap_dist, 100 * ((alpha / 2)))
        val = _np.percentile(bootstrap_dist, 50)
        high = _np.percentile(bootstrap_dist, 100 * (1 - (alpha / 2)))
    return BootstrapResults(low, val, high)


def _needs_sparse_unification(values_lists):
    non_zeros = values_lists[0] != 0

    for v in values_lists:
        v_nz = v != 0
        non_zeros = (non_zeros + v_nz) > 0

    non_zero_size = non_zeros.sum()

    for v in values_lists:
        if non_zero_size != v.data.shape[0]:
            return True

    return False


def _validate_arrays(values_lists):
    t = values_lists[0]
    t_type = type(t)
    if not isinstance(t, _sparse.csr_matrix) and not isinstance(t, _np.ndarray):
        raise ValueError(('The arrays must either be of type '
                          'scipy.sparse.csr_matrix or numpy.array'))

    for _, values in enumerate(values_lists[1:]):
        if not isinstance(values, t_type):
            raise ValueError('The arrays must all be of the same type')

        if t.shape != values.shape:
            raise ValueError('The arrays must all be of the same shape')

        if isinstance(t, _sparse.csr_matrix):
            if values.shape[0] > 1:
                raise ValueError(('The sparse matrix must have shape 1 row X N'
                                  ' columns'))

    if isinstance(t, _sparse.csr_matrix):
        if _needs_sparse_unification(values_lists):
            raise ValueError(('The non-zero entries in the sparse arrays'
                              ' must be aligned: see '
                              'bootstrapped.unify_sparse_vectors function'))


def _generate_distributions(values_lists, num_iterations):
    if isinstance(values_lists[0], _sparse.csr_matrix):
        # in the sparse case we dont actually need to bootstrap
        # the full sparse array since most values are 0
        # instead for each bootstrap iteration we:
        #    1. generate B number of non-zero entries to sample from the
        #          binomial distribution
        #    2. resample with replacement the non-zero entries from values
        #          B times
        #    3. create a new sparse array with the B resamples, zero otherwise
        results = [[] for _ in range(len(values_lists))]

        pop_size = values_lists[0].shape[1]
        non_sparse_size = values_lists[0].data.shape[0]

        p = non_sparse_size * 1.0 / pop_size

        for _ in range(num_iterations):
            ids = _np.random.choice(
                non_sparse_size,
                _np.random.binomial(pop_size, p),
                replace=True,
            )

            for arr, values in zip(results, values_lists):
                data = values.data
                d = _sparse.csr_matrix(
                    (
                        data[ids],
                        (_np.zeros_like(ids), _np.arange(len(ids)))
                    ),
                    shape=(1, pop_size),
                )

                arr.append(d)
        return [_sparse.vstack(r) for r in results]

    else:
        values_shape = values_lists[0].shape[0]
        ids = _np.random.choice(
            values_shape,
            (num_iterations, values_shape),
            replace=True,
        )

        results = [values[ids] for values in values_lists]
        return results


def _bootstrap_sim(values_lists, stat_func_lists, num_iterations,
                   iteration_batch_size, seed):
    '''Returns simulated bootstrap distribution.
    See bootstrap() funciton for arg descriptions.
    '''

    if seed is not None:
        _np.random.seed(seed)

    num_iterations = int(num_iterations)
    iteration_batch_size = int(iteration_batch_size)

    results = [[] for _ in values_lists]

    for rng in range(0, num_iterations, iteration_batch_size):
        max_rng = min(iteration_batch_size, num_iterations - rng)

        values_sims = _generate_distributions(values_lists, max_rng)

        for i, values_sim, stat_func in zip(range(len(values_sims)), values_sims, stat_func_lists):
            results[i].extend([stat_func(i) for i in values_sim])

    return _np.array(results)


def _bootstrap_distribution(values_lists, stat_func_lists,
                            num_iterations, iteration_batch_size, num_threads):
    '''Returns the simulated bootstrap distribution. The idea is to sample the same
        indexes in a bootstrap re-sample across all arrays passed into values_lists.
        This is especially useful when you want to co-sample records in a ratio metric.
            numerator[k].sum() / denominator[k].sum()
        and not
            numerator[ j ].sum() / denominator[k].sum()
    Args:
        values_lists: list of numpy arrays (or scipy.sparse.csr_matrix)
            each represents a set of values to bootstrap. All arrays in values_lists
            must be of the same length.
        stat_func_lists: statistic to bootstrap for each element in values_lists.
        num_iterations: number of bootstrap iterations / resamples / simulations
            to perform.
        iteration_batch_size: The bootstrap sample can generate very large
            matrices. This argument limits the memory footprint by
            batching bootstrap rounds. If unspecified the underlying code
            will produce a matrix of len(values) x num_iterations. If specified
            the code will produce sets of len(values) x iteration_batch_size
            (one at a time) until num_iterations have been simulated.
            Defaults to no batching.
        num_threads: The number of therads to use. This speeds up calculation of
            the bootstrap. Defaults to 1. If -1 is specified then
            multiprocessing.cpu_count() is used instead.
    Returns:
        The set of bootstrap resamples where each stat_function is applied on
        the bootsrapped values.
    '''

    _validate_arrays(values_lists)

    if iteration_batch_size is None:
        iteration_batch_size = num_iterations

    num_iterations = int(num_iterations)
    iteration_batch_size = int(iteration_batch_size)

    num_threads = int(num_threads)

    if num_threads == -1:
        num_threads = _multiprocessing.cpu_count()

    if num_threads <= 1:
        results = _bootstrap_sim(values_lists, stat_func_lists,
                                 num_iterations, iteration_batch_size, None)
    else:
        pool = _multiprocessing.Pool(num_threads)

        iter_per_job = _np.ceil(num_iterations * 1.0 / num_threads)

        results = []
        for seed in _np.random.randint(0, 2 ** 32 - 1, num_threads):
            r = pool.apply_async(_bootstrap_sim, (values_lists, stat_func_lists,
                                                  iter_per_job,
                                                  iteration_batch_size, seed))
            results.append(r)

        results = _np.hstack([res.get() for res in results])

        pool.close()

    return results


def bootstrap(values, stat_func, alpha=0.05,
              num_iterations=1000, iteration_batch_size=10, is_pivotal=True,
              num_threads=1, return_distribution=False):
    '''Returns bootstrap estimate.
    Args:
        values: numpy array of values to bootstrap
        stat_func: statistic to bootstrap. We provide several default functions:
            median_estimator
            method_of_moments_v3_estimator
        alpha: alpha value representing the confidence interval.
            Defaults to 0.05, i.e., 95th-CI.
        num_iterations: number of bootstrap iterations to run. The higher this
            number the more sure you can be about the stability your bootstrap.
            By this - we mean the returned interval should be consistent across
            runs for the same input. This also consumes more memory and makes
            analysis slower. Defaults to 10000.
        iteration_batch_size: The bootstrap sample can generate very large
            matrices. This argument limits the memory footprint by
            batching bootstrap rounds. If unspecified the underlying code
            will produce a matrix of len(values) x num_iterations. If specified
            the code will produce sets of len(values) x iteration_batch_size
            (one at a time) until num_iterations have been simulated.
            Defaults to 10. Passing None will calculate the full simulation in one step.
        is_pivotal: if true, use the pivotal method for bootstrapping confidence
            intervals. If false, use the percentile method.
        num_threads: The number of therads to use. This speeds up calculation of
            the bootstrap. Defaults to 1. If -1 is specified then
            multiprocessing.cpu_count() is used instead.
    Returns:
        BootstrapResults representing CI and estimated value.
    '''

    values_lists = [_np.asarray(values)]
    stat_func_lists = [stat_func]

    stat_val = stat_func(values)

    distribution_results = _bootstrap_distribution(values_lists,
                                                   stat_func_lists,
                                                   num_iterations,
                                                   iteration_batch_size,
                                                   num_threads)

    if return_distribution:
        val = _np.percentile(distribution_results, 50)
        return [i * (stat_val / val) for i in distribution_results]
    else:
        return _get_confidence_interval(distribution_results, stat_val, alpha,
                                        is_pivotal)
