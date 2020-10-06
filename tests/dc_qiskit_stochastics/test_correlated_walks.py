import logging
import unittest
from typing import Tuple, List

import numpy as np

from dc_qiskit_stochastics import benchmark
from dc_qiskit_stochastics.plotting import plot_characteristic_function
from dc_qiskit_stochastics.dsp_correlated_walk import CorrelatedWalk
from dc_quantum_scheduling import processor

logging.basicConfig(format=logging.BASIC_FORMAT, level='INFO')
LOG = logging.getLogger(__name__)


class CorrelatedWalkTests(unittest.TestCase):

    def test_benchmark_brute_force(self):
        probabilities = np.asarray([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        realizations = np.asarray([[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]])

        evaluations = np.arange(0.0, 2 * np.pi, 0.1)

        c_func = lambda e: benchmark.characteristic_function_rw_ind(
            probabilities=probabilities,
            realizations=realizations,
            initial_value=0.0,
            evaluations=[e]
        )

        results: List[Tuple[complex, complex, complex]] = []

        for e in evaluations:

            [c_func_eval_0] = c_func(e)

            c_func_eval_1 = benchmark.brute_force_rw_ind(
                probabilities=probabilities,
                realizations=realizations,
                initial_value=0.0,
                scaling=e,
                func=lambda x: np.exp(1.0j * x)
            )

            c_func_eval_2 = benchmark.brute_force_correlated_walk(
                probabilities=probabilities,
                realizations=realizations,
                initial_value=0.0,
                scaling=e,
                func=lambda x: np.exp(1.0j * x)
            )

            LOG.info(f'Assuming independence. We have: {c_func_eval_0:.4} / {c_func_eval_1:.4} / {c_func_eval_2:.4}')
            results.append((c_func_eval_0, c_func_eval_1, c_func_eval_2))

        arr = np.asarray(results)
        fig = plot_characteristic_function(
            simulation=arr[:, 1],
            experiment=arr[:, 2],
            theory=arr[:,0],
            title=f'benchmark-RW (ind) n={len(probabilities)}'
        )
        fig.show()

        for c0, c1, c2 in results:
            self.assertAlmostEqual(c0.real, c1.real, delta=10e-6)
            self.assertAlmostEqual(c0.imag, c1.imag, delta=10e-6)
            self.assertAlmostEqual(c0.real, c2.real, delta=10e-6)
            self.assertAlmostEqual(c0.imag, c2.imag, delta=10e-6)
            self.assertAlmostEqual(c1.real, c2.real, delta=10e-6)
            self.assertAlmostEqual(c1.imag, c2.imag, delta=10e-6)

    def test_benchmark_monte_carlo(self):
        probabilities = np.asarray([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        realizations = np.asarray([[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]])
        mc_samples = int(1e3)

        evaluations = np.random.uniform(-2 * np.pi, 2 * np.pi, size=5)

        c_func = lambda e: benchmark.characteristic_function_rw_ind(
            probabilities=probabilities,
            realizations=realizations,
            initial_value=0.0,
            evaluations=[e]
        )

        results: List[Tuple[complex, complex, complex]] = []
        for e in evaluations:

            [c_func_eval_0] = c_func(e)

            c_func_eval_1 = benchmark.brute_force_correlated_walk(**{
                    'probabilities': probabilities,
                    'realizations': realizations,
                    'initial_value': 0.0,
                    'scaling': e
                }
            )

            c_func_eval_2 = benchmark.monte_carlo_correlated_walk(**{
                    'probabilities': probabilities,
                    'realizations': realizations,
                    'initial_value': 0.0,
                    'scaling': e,
                    'samples': mc_samples
                }
            )

            LOG.info(f'Assuming independence. We have: {c_func_eval_0:.4} / {c_func_eval_1:.4} / {c_func_eval_2:.4}')
            results.append((c_func_eval_0, c_func_eval_1, c_func_eval_2))

        arr = np.asarray(results)
        fig = plot_characteristic_function(
            simulation=arr[:, 1],
            experiment=arr[:, 2],
            theory=arr[:,0],
            title=f'\nbenchmark-RW (ind/mc) n={len(probabilities)}, s={mc_samples}'
        )
        fig.axes[0].set_ylim([min(-0.1, np.min(np.imag(arr[:, 2])) - 0.01), max(0.1, np.max(np.imag(arr[:, 2]))) + 0.01])
        fig.show()

        for c0, c1, c2 in results:
            self.assertAlmostEqual(c0.real, c1.real, delta=10e-2)
            self.assertAlmostEqual(c0.imag, c1.imag, delta=10e-2)
            self.assertAlmostEqual(c0.real, c2.real, delta=10e-2)
            self.assertAlmostEqual(c0.imag, c2.imag, delta=10e-2)
            self.assertAlmostEqual(c1.real, c2.real, delta=10e-2)
            self.assertAlmostEqual(c1.imag, c2.imag, delta=10e-2)

    def test_correlated_walk_independent(self):

        probabilities = np.asarray([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        realizations = np.asarray([[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]])

        evaluations = np.random.uniform(-2 * np.pi, 2 * np.pi, size=11)
        # evaluations = np.linspace(-2 * np.pi, 2 * np.pi, num=11)

        expected_output: List[complex] = []
        for e in evaluations:
            c_func_eval = benchmark.brute_force_correlated_walk(
                probabilities=probabilities,
                realizations=realizations,
                initial_value=0.0,
                scaling=e,
                func=lambda x: np.exp(1.0j * x)
            )
            expected_output.append(c_func_eval)

        correlated_walk = CorrelatedWalk(
            initial_value=0.0,
            realizations=realizations,
            probabilities=probabilities
        )

        prepared_experiment = correlated_walk.characteristic_function(
            evaluations=evaluations, other_arguments={'shots': 8192*100}
        )

        finished_experiment = processor.execute_simulation(prepared_experiment).wait()
        output = finished_experiment.get_output()

        compare_outputs = list(zip(evaluations, expected_output, output))
        for e, c0, c1 in compare_outputs:
            LOG.info(f'{e:.4}: {c0:.4} / {c1:.4}')

        fig = plot_characteristic_function(output, [], expected_output, title=f'sim-RW (ind) n={len(probabilities)}')
        fig.show()

        for _, c0, c1 in compare_outputs:
            self.assertAlmostEqual(c0.real, c1.real, delta=10e-3)
            self.assertAlmostEqual(c0.imag, c1.imag, delta=10e-3)

    def test_correlated_walk_strong(self):

        probabilities = np.asarray([[0.5, 0.5], [0.8, 0.8], [0.8, 0.8], [0.8, 0.8], [0.8, 0.8]])
        realizations = np.asarray([[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]])

        evaluations = np.random.uniform(-2 * np.pi, 2 * np.pi, size=11)

        expected_output: List[complex] = []
        for e in evaluations:
            c_func_eval = benchmark.brute_force_correlated_walk(
                probabilities=probabilities,
                realizations=realizations,
                initial_value=0.0,
                scaling=e,
                func=lambda x: np.exp(1.0j * x)
            )
            expected_output.append(c_func_eval)

        correlated_walk = CorrelatedWalk(
            initial_value=0.0,
            realizations=realizations,
            probabilities=probabilities
        )

        prepared_experiment = correlated_walk.characteristic_function(
            evaluations=evaluations, other_arguments={'shots': 8192*100}
        )

        finished_experiment = processor.execute_simulation(prepared_experiment).wait()
        output = finished_experiment.get_output()

        compare_outputs = list(zip(evaluations, expected_output, output))
        for e, c0, c1 in compare_outputs:
            LOG.info(f'{e:.4}: {c0:.4} / {c1:.4}')

        fig = plot_characteristic_function(output, [], expected_output, title=f'sim-RW (dep) n={len(probabilities)}')
        fig.show()

        for _, c0, c1 in compare_outputs:
            self.assertAlmostEqual(c0.real, c1.real, delta=10e-3)
            self.assertAlmostEqual(c0.imag, c1.imag, delta=10e-3)
