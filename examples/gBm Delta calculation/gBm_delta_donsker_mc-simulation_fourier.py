import logging
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np

from dc_qiskit_stochastics import benchmark
from gBm_delta_data import S_0, r, time_evaluation, time_of_maturity, time_to_maturity, mu, sigma, K

logging.basicConfig(format=logging.BASIC_FORMAT, level='INFO')
logging.getLogger(__name__).setLevel('INFO')

LOG = logging.getLogger(__name__)

if __name__ == "__main__":

    n = 4

    mc_samples = 10

    P = 10000
    L = 10000

    l_array = np.arange(-L, L + 1)
    fourier_coefficients = np.asarray(
        [-1.0j / (2 * np.pi * l) * np.exp(-2 * np.pi ** 2 * l ** 2 / P ** 2) if l != 0 else 1 / 2
         for l in l_array])
    char_func_input = 2 * np.pi / P * l_array

    strike_prices = np.arange(10, 200, 5)

    with Pool() as pool:
        futures = []
        for strike_price in strike_prices:
            initial_value = (np.log(S_0) - np.log(strike_price) + (r + sigma ** 2 / 2) * time_to_maturity)
            initial_value = initial_value / (sigma * np.sqrt(time_to_maturity))

            x_1 = (mu - sigma ** 2 / 2) / (n * sigma * np.sqrt(time_to_maturity)) \
                - 1 / (np.sqrt(n) * np.sqrt(time_to_maturity))
            x_2 = (mu - sigma ** 2 / 2) / (n * sigma * np.sqrt(time_to_maturity)) \
                + 1 / (np.sqrt(n) * np.sqrt(time_to_maturity))

            future = pool.apply_async(
                benchmark.monte_carlo_rw_ind,
                kwds={
                    'offset': initial_value,
                    'probs': [.5, .5],
                    'choices': [x_1 , x_2],
                    'length': n,
                    'scalings': char_func_input,
                    'mc_samples': mc_samples
                }
            )
            futures.append(future)

        exp_vals = []
        for future in futures:
            char_func_output = np.asarray(future.get())
            exp_val = char_func_output.dot(fourier_coefficients)
            exp_vals.append(exp_val)

    expectation_approx = np.asarray(exp_vals)

    plt.scatter(x=strike_prices, y=expectation_approx)
    plt.title(
        f'gBm: $S_0$={S_0}/K={K}/r={r}/$\mu$={mu}/$\sigma$={sigma}/$t$={time_evaluation},/$T$={time_of_maturity}\n'
        f'RW-MC (Donkser), n={n}, P={P}, L={L}, mc-samples={mc_samples}')
    plt.show()
