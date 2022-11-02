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

import numpy as np
import qiskit
from dc_qiskit_algorithms import MöttönenStatePreparationGate
from dc_qiskit_algorithms.MöttönenStatePreparation import get_alpha_y
from scipy import sparse

LOG = logging.getLogger(__name__)


def index_independent_prep(level: int, probabilities: np.ndarray, **kwargs) -> qiskit.QuantumCircuit:
    """
    The function adds an index register of appropriate size and uses the state preparation by Möttönen et al.
    > Möttönen, Mikko, et al. "Transformation of quantum states using uniformly controlled rotations."
    > Quantum Information & Computation 5.6 (2005): 467-473.

    :param level: The level for register naming purposes
    :param probabilities: the probabilities for this level
    :return: the quantum circuit with the state preparation for the index
    """
    k = probabilities.shape
    qubits_k = int(np.ceil(np.log2(k)))

    qc = qiskit.QuantumCircuit(name='index_state_prep')
    assert np.sum(probabilities) == 1
    qreg = qiskit.QuantumRegister(qubits_k, f'level_{level}')
    qc.add_register(qreg)
    # Möttönen State-prep with row
    qc.append(MöttönenStatePreparationGate(list(np.sqrt(probabilities))), qreg)

    return qc


def index_independent_prep_two(level: int, probabilities: np.ndarray, **kwargs) -> qiskit.QuantumCircuit:
    assert probabilities.shape == (2,)

    qc = qiskit.QuantumCircuit(name='index_state_prep')
    assert np.sum(probabilities) == 1
    qreg = qiskit.QuantumRegister(1, f'level_{level}')
    qc.add_register(qreg)

    vector = sparse.dok_matrix([probabilities]).transpose()
    alpha = get_alpha_y(vector, 1, 1)
    qc.ry(alpha[0, 0], qreg)

    return qc
