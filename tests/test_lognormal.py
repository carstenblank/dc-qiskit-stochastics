import unittest
from multiprocessing import Pool
import numpy as np
from ddt import ddt, data as test_data, unpack
import matplotlib.pyplot as plt
from scipy.special import factorial

from dc_qiskit_stochastics.benchmark import char_func_asian_option
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
        self.assertAlmostEqual(m, 0, delta=1, msg='The distance of the sampled value to the theoretical result is '
                                                  'not met. As this is a randomized algorithm, it may be valid '
                                                  'still. Rerun the test please.')


@ddt
class BenchmarkTests(unittest.TestCase):

    @test_data(
        {
            'risk_free_interest': 0.02,
            'volatility': 0.05,
            'start_value': 80,
            'time_steps': np.asarray([0])
        },
        {
            'risk_free_interest': 0.02,
            'volatility': 0.05,
            'start_value': 80,
            'time_steps': np.asarray([10])
        },
        {
            'risk_free_interest': 0.02,
            'volatility': 0.05,
            'start_value': 50,
            'time_steps': np.asarray([5])
        },
        {
            'risk_free_interest': 0.02,
            'volatility': 0.07,
            'start_value': 20,
            'time_steps': np.asarray([1])
        }
    )
    @unpack
    def test(self, risk_free_interest, volatility, start_value, time_steps):
        mu = (risk_free_interest - 0.5 * volatility ** 2) * time_steps[-1]
        sigma = volatility * np.sqrt(time_steps[-1])

        v = np.linspace(-0.3, 0.3, num=400)

        phi_v_benchmark_m = np.asarray(
            [(1.0j * v * start_value) ** n / factorial(n) * np.exp(n * mu + n**2 * sigma**2 / 2) for n in range(1000)]
        )
        phi_v_benchmark = np.sum(phi_v_benchmark_m, axis=0, where=~np.isnan(phi_v_benchmark_m))

        phi_v = char_func_asian_option(risk_free_interest, volatility, start_value, time_steps, samples=2000, evaluations=v)

        plt.plot(v, np.real(phi_v_benchmark), color='blue', marker='.')
        plt.plot(v, np.imag(phi_v_benchmark), color='orange', marker='.')

        plt.plot(v, np.real(phi_v), color='blue')
        plt.plot(v, np.imag(phi_v), color='orange')

        plt.axvline(x=0, color='blue')
        plt.title(time_steps)
        plt.ylim((-1.1, 1.1))
        plt.show()


# TODO: remove if not needed.
# @ddt
# class BenchmarkTests(unittest.TestCase):
#
#     @test_data(
#         {
#             'risk_free_interest': 0.02,
#             'volatility': 0.05,
#             'start_value': 80,
#             'time_steps': np.asarray([0])
#         },
#         {
#             'risk_free_interest': 0.02,
#             'volatility': 0.05,
#             'start_value': 80,
#             'time_steps': np.asarray([10])
#         },
#         {
#             'risk_free_interest': 0.02,
#             'volatility': 0.05,
#             'start_value': 80,
#             'time_steps': np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#         },
#         {
#             'risk_free_interest': 0.02,
#             'volatility': 0.05,
#             'start_value': 80,
#             'time_steps': np.asarray([1, 4, 7, 10])
#         }
#     )
#     @unpack
#     def test(self, risk_free_interest, volatility, start_value, time_steps):
#         mu = (risk_free_interest - 0.5 * volatility ** 2) * time_steps[-1]
#         sigma = volatility * np.sqrt(time_steps[-1])
#
#         v = np.linspace(-0.3, 0.3, num=400)
#
#         # n = np.arange(100)
#         # phi_v_benchmark= []
#         # for vv in v:
#         #     decay_part = np.exp(n * mu + (n * sigma) ** 2 / 2) / factorial(n)
#         #     oscillating_part = np.asarray([(1.0j * vv * start_value) ** nn for nn in n])
#         #     vec = decay_part * oscillating_part
#         #     value = np.sum([e for e in vec if not np.isnan(e)])
#         #     phi_v_benchmark.append(value)
#         # phi_v_benchmark = np.asarray(phi_v_benchmark)
#
#         phi_v_benchmark_m = np.asarray(
#             [(1.0j * v * start_value) ** n / factorial(n) * np.exp(n * mu + n**2 * sigma**2 / 2) for n in range(1000)]
#         )
#         phi_v_benchmark = np.sum(phi_v_benchmark_m, axis=0, where=~np.isnan(phi_v_benchmark_m))
#
#         phi_v = char_func_asian_option(risk_free_interest, volatility, start_value, time_steps, samples=2000, evaluations=v)
#
#         plt.plot(v, np.real(phi_v_benchmark), color='blue', marker='.')
#         plt.plot(v, np.imag(phi_v_benchmark), color='orange', marker='.')
#
#         plt.plot(v, np.real(phi_v), color='blue')
#         plt.plot(v, np.imag(phi_v), color='orange')
#
#         plt.axvline(x=0, color='blue')
#         plt.title(time_steps)
#         plt.ylim((-1.1, 1.1))
#         plt.show()
