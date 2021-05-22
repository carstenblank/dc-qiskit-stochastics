import unittest
from multiprocessing import Pool
import numpy as np
from ddt import ddt, data as test_data, unpack

from dc_qiskit_stochastics.simulation.lognormal import sample_last


@ddt
class LogNormalTests(unittest.TestCase):

    @test_data(
        {
            'risk_free_interest': 0.02,
            'volatility': 0.05,
            'start_value': 80,
            'time': (0, 10),
            'n': 1
        },
        {
            'risk_free_interest': 0.02,
            'volatility': 0.05,
            'start_value': 80,
            'time': (0, 10),
            'n': 10
        },
        {
            'risk_free_interest': 0.02,
            'volatility': 0.05,
            'start_value': 80,
            'time': (0, 10),
            'n': 20
        }
    )
    @unpack
    def test_full(self, risk_free_interest, volatility, start_value, time, n):

        arguments = [risk_free_interest, volatility, start_value, time, n]
        samples = [arguments] * 100000

        end_value_interest = start_value * np.exp(risk_free_interest * (time[1] - time[0]))

        possible_values = []
        with Pool() as pool:
            for i in range(3):
                print(f'Iteration {i}')
                last = pool.starmap(sample_last, samples)
                possible_values.append(np.average(last))

        possible_values = np.asarray(possible_values)
        print(f'{possible_values}/{end_value_interest}')

        m = np.min(possible_values - end_value_interest)
        print(f'The minimum distance from the correct answer of those iteration was found to be: {m}.')
        self.assertAlmostEqual(m, 0, delta=1e-1, msg='The distance of the sampled value to the theoretical result is '
                                                     'not met. As this is a randomized algorithm, it may be valid '
                                                     'still. Rerun the test please.')
