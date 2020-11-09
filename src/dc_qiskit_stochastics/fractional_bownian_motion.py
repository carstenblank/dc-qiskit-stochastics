import itertools
import logging
import os
import time
from datetime import datetime
from multiprocessing import Pool
from typing import List, Callable, Union, Optional, Tuple

import numpy as np
import scipy
from nptyping import NDArray
from qiskit.ignis.mitigation import CompleteMeasFitter
from qiskit.providers.ibmq import IBMQBackend
from qiskit.transpiler import PassManager
from scipy import sparse
from scipy.special import gamma

from .dsp_correlated_walk import CorrelatedWalk, benchmark_monte_carlo
from .dsp_util import choice
from simulation_scheduling import PreparedExperiment, FinishedExperiment

LOG = logging.getLogger(__name__)


def _benchmark_tuple(a: Tuple) -> np.ndarray:
    logging.basicConfig(format=logging.BASIC_FORMAT, level='INFO')
    LOG.info(f'Starting benchmark #{a[0]}.')
    start = datetime.now()
    result = benchmark_monte_carlo(*a[1:])
    end = datetime.now()
    LOG.info(f'Ended benchmark #{a[0]} in {end - start}.')
    return result


class FractionalBrownianMotion(object):
    density: Callable[[float, float], float]
    hurst_index: float
    approximation_N: int
    approximation_M: int
    time_evaluation: float
    initial_value: float
    walks: List[CorrelatedWalk]
    persistence_list: [List[float] or np.ndarray or NDArray]
    persistence_probabilities_list: [List[float] or np.ndarray]

    def __init__(self, hurst_index: float, initial_value: float, approximation_N: int, approximation_M: int,
                 time_evaluation: float, density: Callable[[float, float], float]):
        if hurst_index < 0.0 or hurst_index > 1.0:
            raise AssertionError("The Hurst needs to be in [0,1].")
        if hurst_index < 0.5:
            raise AssertionError("The Hurst needs to be >= 0.5, values < 0.5 are not implemented yet!")

        self.density = density
        self.hurst_index = hurst_index
        self.time_evaluation = time_evaluation
        self.approximation_M = approximation_M
        self.approximation_N = approximation_N
        self.initial_value: float = initial_value
        self.persistence_list = []
        self.persistence_probabilities_list = []
        self.walks = []

    def _create_correlated_walk(self, persistence: float) -> CorrelatedWalk:
        levels = int(np.ceil(self.approximation_N * self.time_evaluation))
        probabilities = np.asarray([[persistence, persistence] for _ in range(levels)])
        probabilities[0] = [0.5, 0.5]  # the first step is not correlated and is chosen to be equal
        realizations = np.asarray([[-1, 1] for _ in range(levels)])

        return CorrelatedWalk(
            initial_value=self.initial_value,
            probabilities=probabilities,
            realizations=realizations
        )

    def prepare_integral(self, density_descretization: int, debug_plot=False):
        steps, delta = np.linspace(0.5, 1.0, density_descretization, retstep=True)
        density_evaluations = [self.density(self.hurst_index, s) for s in steps]
        density_steps_evals = np.asarray([[s, p] for s, p in zip(steps, density_evaluations) if not np.isinf(p)])
        pdf = delta * density_steps_evals[:, 1]
        steps = density_steps_evals[:, 0]
        assert np.abs(1 - np.sum(pdf)) < 10e-1
        self.persistence_list = steps
        self.persistence_probabilities_list = pdf
        if debug_plot:
            import matplotlib.pyplot as plt
            plt.plot(steps, [self.density(self.hurst_index, s) for s in steps])
            plt.hist(self.persistence_list, density=True, bins=20)
            plt.show()
        self.walks = [self._create_correlated_walk(p) for p in self.persistence_list]

    def sample(self, density_descretization: int, debug_plot=False):
        steps, delta = np.linspace(0.5, 1.0, density_descretization, retstep=True)
        density_evaluations = [self.density(self.hurst_index, s) for s in steps]
        density_steps_evals = np.asarray([[s, p] for s, p in zip(steps, density_evaluations) if not np.isinf(p)])
        pdf = delta * density_steps_evals[:, 1]
        steps = density_steps_evals[:, 0]
        assert np.abs(1 - np.sum(pdf)) < 10e-1
        self.persistence_list = np.asarray(list(choice(steps, pdf, size=self.approximation_M)))
        if debug_plot:
            import matplotlib.pyplot as plt
            plt.plot(steps, [self.density(self.hurst_index, s) for s in steps])
            plt.hist(self.persistence_list, density=True, bins=20)
            plt.show()
        self.walks = [self._create_correlated_walk(p) for p in self.persistence_list]

    def characteristic_function_experiments(self, evaluations: Union[List[float], np.ndarray, scipy.sparse.dok_matrix],
                                external_id: Optional[str] = None, level_func=None, pm: Optional[PassManager] = None,
                                transpiler_target_backend: Optional[Callable[[], IBMQBackend]] = None,
                                other_arguments: dict = None) -> List[PreparedExperiment]:

        if other_arguments.get('force_single_threaded', False):
            return [w.characteristic_function(
                evaluations=evaluations, external_id=external_id, level_func=level_func, pm=pm,
                transpiler_target_backend=transpiler_target_backend, other_arguments=other_arguments
            ) for w in self.walks]

        with Pool(processes=other_arguments.get('mulitprocessing_processes', None)) as pool:
            from dc_qiskit_stochastics.discrete_stochastic_process import DiscreteStochasticProcess
            futures = []
            for w in self.walks:
                future = pool.apply_async(
                    DiscreteStochasticProcess.characteristic_function,
                    args=(w,),
                    kwds={
                        'evaluations': evaluations, 'external_id': external_id, 'level_func': level_func, 'pm': pm,
                        'transpiler_target_backend': transpiler_target_backend, 'other_arguments': other_arguments
                    }
                )
                futures.append(future)
            readied_list = [f.ready() for f in futures]
            while not all(readied_list):
                LOG.info(f'Still waiting for {len([r for r in readied_list if r == False])} '
                         f'of {len(futures)} experiments to be prepared.')
                time.sleep(5)
                readied_list = [f.ready() for f in futures]
            LOG.info(f'Done with preparing all {len(readied_list)} experiments.')
            return [f.get() for f in futures]

    def characteristic_function(self, experiments: List[FinishedExperiment], fitter_experiments: List[FinishedExperiment]):
        measure_fitter_list: List[CompleteMeasFitter] = [e.get_output() for e in fitter_experiments]
        results: List[np.ndarray] = [e.get_output(fitter) for e, fitter in
                                     itertools.zip_longest(experiments, measure_fitter_list, fillvalue=None)]
        result_matrix = np.asarray(results)
        return self.output(result_matrix)

    def _get_factor(self):
        if self.hurst_index == 0.5:
            c_H = 1/np.sqrt(2)
            denominator = np.sqrt(self.approximation_N * np.log(self.approximation_N)) * np.sqrt(self.approximation_M)
        else:
            c_H = np.sqrt(self.hurst_index * (2 * self.hurst_index - 1)/gamma(3 - 2 * self.hurst_index))
            denominator = self.approximation_N**self.hurst_index * np.sqrt(self.approximation_M)
        factor = c_H / denominator
        return factor

    def output_old(self, matrix: Union[NDArray, np.ndarray]):
        assert len(matrix.shape) == 2
        # Each column is one evaluation, each row is another sampled persistence
        result: List[complex] = []
        for column in range(matrix.shape[1]):
            evaluations_per_persistence = matrix[:, column]
            if column in self.persistence_probabilities_list:
                probability_of_persistence = self.persistence_probabilities_list[column]
            else:
                probability_of_persistence = 1
            evaluations_per_persistence = probability_of_persistence * evaluations_per_persistence
            summed_up = np.product(evaluations_per_persistence)
            result.append(summed_up**self.approximation_M)
        return np.asarray(result)

    def output(self, matrix: Union[NDArray, np.ndarray]):
        assert len(matrix.shape) == 2
        # Each column is one evaluation, each row is another sampled persistence
        result: List[complex] = []
        for column in range(matrix.shape[1]):
            evaluations_per_persistence = matrix[:, column]
            if column in self.persistence_probabilities_list:
                probability_of_persistence = self.persistence_probabilities_list[column]
            else:
                probability_of_persistence = 0
            evaluations_per_persistence = probability_of_persistence * evaluations_per_persistence
            summed_up = np.sum(evaluations_per_persistence)
            result.append(summed_up)
        return np.asarray(result)

    def benchmark(self, evaluations: Union[List[float], np.ndarray, scipy.sparse.dok_matrix], func=None,
                  other_arguments: Optional[dict] =None) -> Union[NDArray[complex], np.ndarray]:
        other_arguments = {} if other_arguments is None else other_arguments
        samples = other_arguments.get('monte_carlo_samples', 100)

        evaluations = np.asarray(evaluations)
        evaluations_scaled = np.vectorize(lambda v: v * self._get_factor())(evaluations)

        if other_arguments.get('force_single_threaded', False):
            matrix = np.asarray(
                [
                    _benchmark_tuple((i, w.probabilities, w.realizations, evaluations_scaled, w.initial_value, samples, func))
                    for i, w in enumerate(self.walks)
                ]
            )
        else:
            processes = other_arguments.get('mulitprocessing_processes', os.cpu_count())
            with Pool(processes=processes) as pool:
                matrix = np.asarray(
                    pool.map(
                        _benchmark_tuple,
                        [(i, w.probabilities, w.realizations, evaluations_scaled, w.initial_value, samples, func)
                         for i, w in enumerate(self.walks)],
                        chunksize=int(len(self.walks) / processes)
                    )
                )
        return self.output(matrix)
