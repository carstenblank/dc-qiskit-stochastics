import logging
import unittest
from typing import List, Union

import numpy as np
from ddt import ddt, data as test_data, unpack

from dc_qiskit_stochastics.discrete_stochastic_process import DiscreteStochasticProcess
from dc_qiskit_stochastics.dsp_common import apply_level_two_realizations
from dc_quantum_scheduling import FinishedExperiment, RunningExperiment, PreparedExperiment, processor

logging.basicConfig(format=logging.BASIC_FORMAT, level='DEBUG')
LOG = logging.getLogger(__name__)


def characteristic_function_two_step(initial_value: float, probabilities: np.ndarray, realizations: np.ndarray):
    def eval(p: float, r1: float, r2: float):
        return p * np.exp(1.0j * r1) + (1 - p) * np.exp((1.0j * r2))

    def _inner(evaluations: Union[float, List[float], np.ndarray]):
        if isinstance(evaluations, float):
            evaluations = np.asarray([evaluations])
        if isinstance(evaluations, list):
            evaluations = np.asarray(evaluations)
        outcomes = []
        for v in evaluations:
            single_char_func_eval_list = [eval(1.0, v * initial_value, 0.0)]
            for p, r in zip(list(probabilities), list(v * realizations)):
                value = eval(p[0], r[0], r[1])
                single_char_func_eval_list.append(value)

            outcomes.append(np.prod(single_char_func_eval_list))
        return outcomes

    return _inner


@ddt
class Execution(unittest.TestCase):

    @test_data(
        ([0.1, 0.2, 0.3], 0.0, np.asarray(4*[[.5, .5]]), np.asarray(4*[[-1, 1]]), apply_level_two_realizations),
        ([0.1, 0.2, 0.3], 5.0, np.asarray(4*[[.5, .5]]), np.asarray(4*[[-1, 1]]), apply_level_two_realizations),
        ([0.1, 0.2, 0.3], -5.0, np.asarray(4*[[.5, .5]]), np.asarray(4*[[-1, 1]]), apply_level_two_realizations),
        ([0.1, 0.2, 0.3], -5.0, np.asarray(4*[[.5, .5]]), np.asarray([[-1, 1], [-0.5, 1], [-0.1, 0.3], [0.0, 2.5]]), apply_level_two_realizations),
        ([0.1, 0.2, 0.3], -5.0, np.asarray([[.5, .5], [.1, .9], [.3, .7], [.8, .2]]), np.asarray(4*[[-1, 1]]), apply_level_two_realizations),
        ([0.1, 0.2, 0.3], -5.0, np.asarray([[.5, .5], [.1, .9], [.3, .7], [.8, .2]]), np.asarray([[-1, 1], [-0.5, 1], [-0.1, 0.3], [0.0, 2.5]]), apply_level_two_realizations),
    )
    @unpack
    def test_calculate_y_measurement_sin(self, values: List[float], initial_value: float, probabilities: np.ndarray,
                                         realizations: np.ndarray, apply_func):

        # output_expected = np.asarray([np.cos(s) ** 4 for s in values])
        func = characteristic_function_two_step(initial_value, probabilities, realizations)
        output_expected = func(values)
        LOG.info(f'Result Expected: {output_expected}')

        dsp: DiscreteStochasticProcess = DiscreteStochasticProcess(initial_value=initial_value, probabilities=probabilities,
                                        realizations=realizations)

        prepared_experiment: PreparedExperiment = dsp.characteristic_function(values, level_func=apply_func,
                                                                              other_arguments={'shots': 2**20})
        running_simulation: RunningExperiment = processor.execute_simulation(prepared_experiment)
        finished_simulation: FinishedExperiment = running_simulation.wait()

        output = finished_simulation.get_output()

        LOG.info(f'Starting Assertions on output: {output}')
        comparison = list(zip(output, output_expected))
        for actual, expected in comparison:
            LOG.info(f'Comparing: {actual} vs. {expected}')
            self.assertAlmostEqual(actual.real, expected.real, delta=10e-3)
            self.assertAlmostEqual(actual.imag, expected.imag, delta=10e-3)
