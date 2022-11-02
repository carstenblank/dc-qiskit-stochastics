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
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gBm_delta_data import S_0, r, time_evaluation, time_of_maturity, time_to_maturity, mu, sigma, K

logging.basicConfig(format=logging.BASIC_FORMAT, level='INFO')
logging.getLogger(__name__).setLevel('INFO')

LOG = logging.getLogger(__name__)


def function(S_0, mu, sigma, r, K, t, T):
    time_to_maturity = T - t

    def to_gBm(t, b):
        return S_0 * np.exp((mu - sigma ** 2 / 2) * t + sigma * b)

    def _pre_delta(x):
        inner_value = (np.log(x / K) + (r + sigma ** 2 / 2) * time_to_maturity) / (
                sigma * np.sqrt(time_to_maturity))
        return np.exp(1.0j * inner_value)

    def _inner(x):
        return _pre_delta(to_gBm(t, x))

    return _inner


if __name__ == "__main__":

    # n = 4
    n = 10

    hurst = 0.5

    # P = 100
    P = 10000
    # L = 100
    L = 10000

    l_array = np.arange(-L, L + 1, 1)

    def calc(strike_prices: np.ndarray):
        exp_vals = []
        for strike_price in strike_prices:
            parts = []
            for l in l_array:
                if l == 0:
                    continue
                x_0 = np.log(S_0) - np.log(strike_price) + (mu - sigma**2 / 2) * time_evaluation + (r + sigma**2/2) * time_to_maturity
                x_0 = x_0 / (sigma * np.sqrt(time_to_maturity))

                inner = - (2 * np.pi * l / P)**2 * (1 + 1/(2 * time_to_maturity**2)) + 1.0j * 2 * np.pi * l / P * x_0

                part = 1.0j / (2 * np.pi * l) * np.exp(inner)
                parts.append(part)
            exp_val = 1/2 - np.sum(parts)
            exp_vals.append(exp_val)
        return np.round(np.asarray(exp_vals), decimals=12)

    strike_prices = np.arange(10, 225, 5)
    expectation_approx = calc(strike_prices)

    plt.scatter(x=strike_prices, y=expectation_approx)
    plt.title(f'gBm: $S_0$={S_0}/K={K}/r={r}/$\mu$={mu}/$\sigma$={sigma}/$t$={time_evaluation},/$T$={time_of_maturity}\n'
              f'underlying fBm hurst={hurst}, P={P}, L={L}')
    plt.show()

    df = pd.DataFrame(data=zip(strike_prices, np.real(expectation_approx), np.imag(expectation_approx)),
                      columns=['strike price', 'expectation value (real)', 'expectation value (imag)'])
    if os.path.exists("./data"):
        df.to_csv(f'./data/gBm-delta-theory-fourier-P={P}-L={L}-n={n}.csv')
