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
from multiprocessing import Pool

import dill
import matplotlib.pyplot as plt
import numpy as np

from dc_qiskit_stochastics import benchmark
from gBm_delta_data import S_0, r, time_evaluation, time_of_maturity, time_to_maturity, mu, sigma, K

logging.basicConfig(format=logging.BASIC_FORMAT, level='INFO')
logging.getLogger(__name__).setLevel('INFO')

LOG = logging.getLogger(__name__)


if __name__ == "__main__":

    n = 10

    mc_samples = 50

    P = 1000
    L = 1000

    l_array = np.arange(-L, L + 1)
    fourier_coefficients = np.asarray(
        [-1.0j / (2 * np.pi * l) * np.exp(-2 * np.pi ** 2 * l ** 2 / P ** 2) if l != 0 else 1 / 2 for l in l_array]
    )
    char_func_input = 2 * np.pi / P * l_array

    strike_prices = np.arange(10, 225, 1)

    with Pool(processes=16) as pool:
        futures = {}
        for strike_price in strike_prices:
            LOG.info(f'Calculation of strike price {strike_price}...')
            initial_value = (np.log(S_0) - np.log(strike_price) + (r + sigma ** 2 / 2) * time_to_maturity)
            initial_value = initial_value / (sigma * np.sqrt(time_to_maturity))

            x_1 = (mu - sigma ** 2 / 2) / (n * sigma * np.sqrt(time_to_maturity)) \
                - 1 / (np.sqrt(n) * np.sqrt(time_to_maturity))
            x_2 = (mu - sigma ** 2 / 2) / (n * sigma * np.sqrt(time_to_maturity)) \
                + 1 / (np.sqrt(n) * np.sqrt(time_to_maturity))
            kwds = {
                'initial_value': initial_value,
                'probabilities': np.asarray([[0.5, 0.5]] + (n - 1) * [[0.9, 0.9]]),
                # 'probabilities': np.asarray([[0.5, 0.5]] + (n - 1) * [[0.5, 0.5]]),
                # 'probabilities': np.stack(
                #     # (np.linspace(0.5, 0.5, n), np.linspace(0.5, 0.5, n)), axis=-1
                #     (np.linspace(0.5, 1.0, n), np.linspace(0.5, 0.6, n)), axis=-1
                # ),
                'realizations': np.asarray(n * [[x_1, x_2]]),
                'scaling': 1,
                # 'func': dill.dumps(lambda x: np.exp(1.0j * char_func_input * x))
                'func': dill.dumps(lambda x: np.asarray([x, x**2]))
                # 'samples': mc_samples
            }
            future = pool.apply_async(benchmark.brute_force_correlated_walk, kwds=kwds)
            # future = pool.apply_async(benchmark.monte_carlo_correlated_walk, kwds=kwds)
            LOG.info(f'Future for strike price {strike_price} added.')
            futures[strike_price] = future

        exp_vals = {}
        for strike_price, future in futures.items():
            LOG.info(f'Receiving future for strike price {strike_price}...')
            char_func_output = np.asarray(future.get())
            exp_val = char_func_output.dot(fourier_coefficients)
            exp_vals[strike_price] = exp_val

    expectation_approx = np.asarray(list(exp_vals.items()))

    plt.plot(expectation_approx[:, 0], np.real(expectation_approx[:, 1]))
    plt.plot(expectation_approx[:, 0], np.imag(expectation_approx[:, 1]))

    plt.title(
        f'gBm: $S_0$={S_0}/K={K}/r={r}/$\mu$={mu}/$\sigma$={sigma}/$t$={time_evaluation},/$T$={time_of_maturity}\n'
        f'CRW-BF[p=0.9] (Donkser), n={n}, P={P}, L={L}')
        # f'RW-BF (Donkser), n={n}, P={P}, L={L}')
    plt.show()
