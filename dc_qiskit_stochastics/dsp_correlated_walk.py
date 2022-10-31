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
from typing import List, Union

import numpy as np
import qiskit
import scipy
from dc_qiskit_algorithms.MöttönenStatePreparation import get_alpha_y
from qiskit.circuit import Parameter
from scipy import sparse

from .benchmark import brute_force_correlated_walk, monte_carlo_correlated_walk
from .discrete_stochastic_process import DiscreteStochasticProcess

LOG = logging.getLogger(__name__)


def index_binary_correlated_walk(level: int, level_p: np.ndarray, **kwargs) -> qiskit.QuantumCircuit:
    probs = []
    probs.append([np.sqrt(level_p[0]), np.sqrt(1 - level_p[0])])
    probs.append([np.sqrt(1 - level_p[1]), np.sqrt(level_p[1])])

    qc = qiskit.QuantumCircuit(name='index_state_prep')
    qreg = qiskit.QuantumRegister(1, f'level_{level}')
    qc.add_register(qreg)

    if level == 0:
        vector = sparse.dok_matrix([probs[0]]).transpose()
        alpha = get_alpha_y(vector, 1, 1)
        qc.ry(alpha[0, 0], qreg)
    else:
        previous_level_qreg = qiskit.QuantumRegister(1, f'level_{level - 1}')
        qc.add_register(previous_level_qreg)

        # Activating the 1 path
        vector = sparse.dok_matrix([probs[1]]).transpose()
        alpha = get_alpha_y(vector, 1, 1)
        qc.cry(alpha[0, 0], previous_level_qreg, qreg)

        # Activating the 0 path
        vector = sparse.dok_matrix([probs[0]]).transpose()
        alpha = get_alpha_y(vector, 1, 1)
        qc.x(previous_level_qreg)
        qc.cry(alpha[0, 0], previous_level_qreg, qreg)
        qc.x(previous_level_qreg)

    return qc


def benchmark_brute_force(probabilities,
              realizations,
              evaluations: Union[List[float], np.ndarray, scipy.sparse.dok_matrix],
              func=None) -> np.ndarray:
    logging.basicConfig(format=logging.BASIC_FORMAT)

    output: List[complex] = []
    for e in list(evaluations):
        c_func_eval = brute_force_correlated_walk(
            probabilities=probabilities,
            realizations=realizations,
            initial_value=0.0,
            scaling=e,
            func=lambda x: np.exp(1.0j * x) if func is None else func
        )
        output.append(c_func_eval)
    return np.asarray(output)


def benchmark_monte_carlo(
        probabilities,
        realizations,
        evaluations: Union[List[float], np.ndarray, scipy.sparse.dok_matrix],
        initial_value: float = 0.0,
        samples: int = 100,
        func=None) -> np.ndarray:
    c_func_eval = monte_carlo_correlated_walk(
        probabilities=probabilities,
        realizations=realizations,
        initial_value=initial_value,
        samples=samples,
        scaling=evaluations,
        func=lambda v: np.exp(1.0j * v) if func is None else func
    )

    return np.asarray(c_func_eval)


class CorrelatedWalk(DiscreteStochasticProcess):
    def __init__(self, initial_value: float, probabilities: np.ndarray, realizations: np.ndarray):
        assert probabilities.shape == realizations.shape
        super().__init__(initial_value, probabilities, realizations)

    def _proposition_one_circuit(self, scaling: Parameter, level_func=None, index_state_prep=None, **kwargs):
        index_state_prep = index_binary_correlated_walk if index_state_prep is None else index_state_prep
        return super(CorrelatedWalk, self)._proposition_one_circuit(scaling, level_func, index_state_prep, **kwargs)

    def benchmark(self, evaluations: Union[List[float], np.ndarray, scipy.sparse.dok_matrix],
                  func=None, samples: int = 100) -> np.ndarray:
        return benchmark_monte_carlo(
            probabilities=self.probabilities,
            realizations=self.realizations,
            evaluations=evaluations,
            initial_value=self.initial_value,
            samples=samples,
            func=func
        )
