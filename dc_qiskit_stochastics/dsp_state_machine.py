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
from typing import Tuple

import numpy as np
import qiskit
from qiskit.circuit import Parameter

from dc_qiskit_algorithms import MöttönenStatePreparationGate
from dc_qiskit_algorithms import ControlledStatePreparationGate
from scipy import sparse

from .discrete_stochastic_process import DiscreteStochasticProcess
from .dsp_common import apply_level

LOG = logging.getLogger(__name__)


def _index_prep_level_0(probabilities: np.ndarray, **kwargs) -> qiskit.QuantumCircuit:
    """
    The function adds an index register of appropriate size and uses the state preparation by Möttönen et al.
    > Möttönen, Mikko, et al. "Transformation of quantum states using uniformly controlled rotations."
    > Quantum Information & Computation 5.6 (2005): 467-473.

    :param level: The level for register naming purposes
    :param probabilities: the probabilities for this level
    :return: the quantum circuit with the state preparation for the index
    """
    _, m = probabilities.shape
    qubits_target = int(np.ceil(np.log2(m)))

    # check if we have a probability density
    assert probabilities.shape[0] == 1
    assert np.linalg.norm(np.sum(probabilities) - 1) < 1e-6, "The vector's entries must sum to 1"

    qc = qiskit.QuantumCircuit(name='index_state_prep')

    qreg_current = qiskit.QuantumRegister(qubits_target, f'level_0')
    qc.add_register(qreg_current)

    # Möttönen State-prep with row
    amplitudes = np.sqrt(probabilities)
    vector = sparse.dok_matrix(amplitudes).transpose()
    gate = MöttönenStatePreparationGate(vector, neglect_absolute_value=False)
    qc.append(gate, list(qreg_current), [])

    return qc


def _index_prep(level: int, probabilities: np.ndarray, with_debug_circuit: bool = False, **kwargs) -> qiskit.QuantumCircuit:
    """
    The function adds an index register of appropriate size and uses the state preparation by Möttönen et al.
    > Möttönen, Mikko, et al. "Transformation of quantum states using uniformly controlled rotations."
    > Quantum Information & Computation 5.6 (2005): 467-473.

    :param level: The level for register naming purposes
    :param probabilities: the probabilities for this level
    :return: the quantum circuit with the state preparation for the index
    """
    assert level > 0, "This call is only for the level 1, and beyond."

    n, m = probabilities.shape
    qubits_source = int(np.ceil(np.log2(n)))
    qubits_target = int(np.ceil(np.log2(m)))

    # check if we have a stochastic matrix
    # assert np.linalg.norm(np.sum(probabilities, axis=0) - 1) < 1e-6, "The probability matrix must be a stochastic matrix"
    assert np.linalg.norm(np.sum(probabilities, axis=1) - 1) < 1e-6, "The probability matrix must be a stochastic matrix"

    qc = qiskit.QuantumCircuit(name='index_state_prep')

    qreg_last = qiskit.QuantumRegister(qubits_source, f'level_{level - 1}')
    qreg_current = qiskit.QuantumRegister(qubits_target, f'level_{level}')
    qc.add_register(qreg_last)
    qc.add_register(qreg_current)

    # Möttönen State-prep with row
    amplitudes = probabilities ** 0.5
    matrix = sparse.dok_matrix(amplitudes)
    gate = ControlledStatePreparationGate(matrix)

    if with_debug_circuit:
        from qiskit import QuantumCircuit
        qc_def: QuantumCircuit = gate.definition
        qc = qc.compose(qc_def, qubits=list(qreg_last) + list(qreg_current))
    else:
        qc.append(gate, list(qreg_last) + list(qreg_current), [])

    return qc


def _index(level, probabilities, **kwargs):
    # Deliver the index register which encodes the joint probability of realizations
    # Given the initial value, select the row that is encoding the first step
    level_probabilities = np.asarray(probabilities)
    if level == 0:
        return _index_prep_level_0(level_probabilities, **kwargs)
    else:
        return _index_prep(level, level_probabilities, **kwargs)


def _level(level, realizations, scaling, **kwargs):
    # Deliver the index register which encodes the joint probability of realizations
    # Given the initial value, select the row that is encoding the first step
    return apply_level(level, realizations, scaling, **kwargs)


class StateMachineDSP(DiscreteStochasticProcess):

    def __init__(self, initial_value: float, probabilities: np.ndarray, realizations: np.ndarray):
        # FIXME: this is wrong, find assertions that reflect this situation
        # assert probabilities.shape == realizations.shape

        _probabilities = np.asarray([np.asarray(p) for p in probabilities])
        _realizations = np.asarray([np.asarray(r) for r in realizations])

        super().__init__(initial_value, _probabilities, _realizations)
        last_target_states = -1
        for step in range(self.length):
            source_states, target_states = self.number_of_states(step)
            if last_target_states != -1:
                assert source_states == last_target_states
            last_target_states = target_states

    @property
    def length(self):
        return self.probabilities.shape[0]

    def number_of_states(self, step: int) -> Tuple[int, ...]:
        return np.asarray(self.probabilities[step]).shape

    def get_level_transition_matrix(self, level: int) -> np.ndarray:
        return self.probabilities[level]

    def _proposition_one_circuit(self, scaling: Parameter, level_func=None, index_state_prep=None, **kwargs):
        index_state_prep = _index if index_state_prep is None else index_state_prep
        level_func = _level if level_func is None else level_func
        return super(StateMachineDSP, self)._proposition_one_circuit(scaling, level_func, index_state_prep, **kwargs)

    #
    # def benchmark(self, evaluations: Union[List[float], np.ndarray, scipy.sparse.dok_matrix],
    #               func=None, samples: int = 100) -> Union[NDArray[complex], np.ndarray]:
    #     return benchmark_monte_carlo(
    #         probabilities=self.probabilities,
    #         realizations=self.realizations,
    #         evaluations=evaluations,
    #         initial_value=self.initial_value,
    #         samples=samples,
    #         func=func
    #     )
