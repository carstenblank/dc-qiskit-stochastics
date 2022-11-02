# Copyright 2018-2022 Carsten Blank
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging

import numpy as np

from dc_qiskit_stochastics import benchmark, dft

LOG = logging.getLogger(__name__)


def step_function(t: np.ndarray, x):
    return (t <= x).astype(int)


def get_statistics(initial_value, realizations, probabilities):
    expectation_value = benchmark.brute_force_correlated_walk(
        probabilities=probabilities,
        realizations=realizations,
        initial_value=initial_value,
        scaling=1,
        func=lambda x: x
    )
    second_moment = benchmark.brute_force_correlated_walk(
        probabilities=probabilities,
        realizations=realizations,
        initial_value=initial_value,
        scaling=1,
        func=lambda x: x**2
    )
    std_deviation = np.sqrt(second_moment - expectation_value**2)

    kurtosis = benchmark.brute_force_correlated_walk(
        probabilities=probabilities,
        realizations=realizations,
        initial_value=initial_value,
        scaling=1,
        func=lambda x: ((x - expectation_value)/second_moment)**4
    )
    LOG.debug(f'Data: initial value: {initial_value}, x_1: {list(realizations[:, 0])} and x_2: {list(realizations[:, 1])} '
             f'with persistence h: {list(probabilities[:,0])}, l: {list(probabilities[:,1])}')
    LOG.info(F'Expectation Value: {expectation_value} and Std Deviation: {std_deviation}, kurtosis: {kurtosis}')
    return expectation_value, std_deviation, kurtosis


def bisection_var(alpha, x_high, x_low, cdf_func, **kwargs):
    cdf = 1.0
    x = None
    while abs(cdf - alpha) > 1e-8 and x_high - x_low > 1e-10:
        x = 0.75 * x_high + 0.25 * x_low
        cdf = cdf_func(x, **kwargs)
        if cdf < alpha:
            x_low = x
        else:
            x_high = x
    return x_high


def bisection_var_fourier(x, discretization, char_func):
    function = step_function(discretization, x)
    coefficients = dft.get_coefficients(function)
    return np.real(coefficients.dot(char_func))


def bisection_var_sim(alpha, probabilities, realizations, initial_value, mc_samples, sampling_func):
    x_high = initial_value + sum(max(r) for r in realizations) + 1
    x_low = initial_value + sum(min(r) for r in realizations)

    def kwds(x):
        return{
            'initial_value': initial_value,
            'probabilities': probabilities,
            'realizations': realizations,
            'scaling': 1,
            'func': lambda X: step_function(X, x),
            'samples': mc_samples
        }

    return bisection_var(alpha, x_high, x_low, cdf_func=lambda X: sampling_func(**kwds(X)))
