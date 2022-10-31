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
import bisect
import logging
from multiprocessing import Pool

from numpy.random import random
import sys
from typing import List, Union, Optional, Dict, Tuple

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.ignis.mitigation import CompleteMeasFitter
from qiskit.providers import BaseBackend
from qiskit.providers.ibmq import IBMQBackend
from qiskit.providers.models import QasmBackendConfiguration
from qiskit.qobj import Qobj
from qiskit.result import Result
from qiskit.transpiler import PassManager

from . import qobj_mapping
from dc_quantum_scheduling import FinishedExperiment

LOG = logging.getLogger(__name__)


def _bind(qc, parameter, v):
    return qc.bind_parameters({parameter: v})


def create_qobj(qc_cos: QuantumCircuit, qc_sin: QuantumCircuit,
                parameter: Parameter, evaluations: np.ndarray,
                qobj_id: str, pass_manager: PassManager,
                other_arguments: dict,
                transpiler_target_backend: Union[BaseBackend, IBMQBackend],
                pre_pass_manager: Optional[PassManager] = None) -> List[Qobj]:
    LOG.info(f'Transpiling {len(evaluations)} cosine and {len(evaluations)} sine circuits with id={qobj_id}.')
    # We have the same number of circuits for the cosine and sine experiments

    other_arguments = {} if other_arguments is None else other_arguments

    if pre_pass_manager:
        LOG.info(f'Pre running cicuits...')
        qc_cos = pre_pass_manager.run(qc_cos)
        qc_sin = pre_pass_manager.run(qc_sin)

    LOG.info(f'Main run of cicuits...')
    qc_transpiled_cos_param = pass_manager.run(qc_cos)
    qc_transpiled_sin_param = pass_manager.run(qc_sin)

    with Pool() as pool:
        circuits_cos = pool.starmap(_bind, [(qc_transpiled_cos_param, parameter, v) for v in evaluations])
        circuits_sin = pool.starmap(_bind, [(qc_transpiled_sin_param, parameter, v) for v in evaluations])

    # circuits_cos = [qc_transpiled_cos_param.bind_parameters({parameter: v}) for v in evaluations]
    # circuits_sin = [qc_transpiled_sin_param.bind_parameters({parameter: v}) for v in evaluations]

    config: QasmBackendConfiguration = transpiler_target_backend.configuration()
    max_shots = config.max_shots
    max_experiments = config.max_experiments if hasattr(config, 'max_experiments') else sys.maxsize
    shots = other_arguments.get('shots', max_shots)
    other_arguments['shots'] = shots

    mapping_matrix: np.array = qobj_mapping(shots, max_shots, max_experiments, 2*len(evaluations))
    shots_per_experiment = int(shots / mapping_matrix.shape[1])
    LOG.debug(mapping_matrix)
    number_of_qobj: int = np.max(mapping_matrix) + 1

    LOG.info(f'For #{len(evaluations)} evaluations each {shots} on the device {transpiler_target_backend.name()} '
             f'with max shots {max_shots} and max no. of experiments per Qobj {max_experiments} '
             f'there are #{number_of_qobj} of Qobj needed. Assembling the circuits now.')

    all_circuits = list(zip(circuits_cos, circuits_sin))

    qobj_list: List[Qobj] = []
    circuit_list: List[List[QuantumCircuit]] = [[] for i in range(number_of_qobj)]

    for row_no, (ccos, csin) in enumerate(all_circuits):
        # cosine
        indices = mapping_matrix[2 * row_no]
        for i, qobj_id in enumerate(indices):
            qobj_circuits = circuit_list[qobj_id]
            qobj_circuits.append(ccos)
        # sine
        indices = mapping_matrix[2 * row_no + 1]
        for i, qobj_id in enumerate(indices):
            qobj_circuits = circuit_list[qobj_id]
            qobj_circuits.append(csin)

    LOG.info(f'Assembled circuits, now building a list of Qobj.')

    for i, cc in enumerate(circuit_list):
        qobj_id_i = f'{qobj_id}--{i}'
        LOG.info(f'Assembling circuits for Qobj #{i} with shots {shots_per_experiment} and id {qobj_id_i}.')
        qobj = qiskit.compiler.assemble(cc,
                                        shots=shots_per_experiment,
                                        max_credits=shots_per_experiment * 5,
                                        qobj_id=qobj_id_i)
        qobj_list.append(qobj)

    return qobj_list


def get_row_colums_of_qobj_index(qobj_index: int, mapping: np.array):
    indices: List[Tuple[int, int]] = []
    for row_no, row in enumerate(mapping):
        for column_no, entry in enumerate(row):
            if entry == qobj_index:
                indices.append((row_no, column_no))
            if entry > qobj_index:
                break
    return indices


def _get_expval_proposition_one(counts: Dict[str, int]):
    return (counts.get('0', 0) - counts.get('1', 0)) / (counts.get('0', 0) + counts.get('1', 0))


def extract_evaluations(finished_experiment: FinishedExperiment, meas_fitter: Optional[CompleteMeasFitter] = None) -> np.ndarray:
    backend: BaseBackend = finished_experiment.transpiler_backend

    config: QasmBackendConfiguration = backend.configuration()
    max_shots = config.max_shots
    max_experiments = config.max_experiments if hasattr(config, 'max_experiments') else sys.maxsize
    shots = finished_experiment.parameters.get('shots', max_shots)

    mapping_matrix = qobj_mapping(shots, max_shots, max_experiments, 2*len(finished_experiment.arguments))
    number_of_qobj: int = np.max(mapping_matrix) + 1

    counts_cos: List[Dict[str, int]] = [{'0': 0, '1': 0} for a in finished_experiment.arguments]
    counts_sin: List[Dict[str, int]] = [{'0': 0, '1': 0} for a in finished_experiment.arguments]

    for qobj_index in range(number_of_qobj):
        indices = get_row_colums_of_qobj_index(qobj_index, mapping_matrix)
        result: Result = finished_experiment.to_result(qobj_index)
        measured_qubits = [e.instructions[-1].qubits for e in finished_experiment.qobj_list[qobj_index].experiments]

        def mitigate_counts(counts: Dict[str, int], q_no: List[int]):
            if meas_fitter is not None:
                # build a fitter from the subset
                meas_fitter_sub = meas_fitter.subset_fitter(qubit_sublist=q_no)
                # Get the filter object
                meas_filter = meas_fitter_sub.filter
                # Results with mitigation
                mitigated_counts = meas_filter.apply(counts, method='pseudo_inverse')
                return mitigated_counts
            else:
                return counts

        count: Dict[str, int]
        for count, (row_no, _), measured_q_no in zip(result.get_counts(), indices, measured_qubits):
            count = mitigate_counts(count, measured_q_no)
            if row_no % 2 == 1:
                argument_no = int((row_no - 1) / 2)
                counts = counts_sin
            else:
                argument_no = int(row_no / 2)
                counts = counts_cos

            old_counts = counts[argument_no]
            counts[argument_no]['0'] = old_counts.get('0', 0) + count.get('0', 0)
            counts[argument_no]['1'] = old_counts.get('1', 0) + count.get('1', 0)

    cosine_expvals = [_get_expval_proposition_one(c) for c in counts_cos]
    sine_expvals = [_get_expval_proposition_one(c) for c in counts_sin]

    return np.asarray([cos + 1.0j * sin for cos, sin in zip(cosine_expvals, sine_expvals)])


def cdf(weights):
    """
    From https://stackoverflow.com/questions/4113307/pythonic-way-to-select-list-elements-with-different-probability
    :param weights:
    :return:
    """
    total = sum(weights)
    result = []
    cumsum = 0
    for w in weights:
        cumsum += w
        result.append(cumsum / total)
    return result


def choice(population, weights, size=None):
    """
    From https://stackoverflow.com/questions/4113307/pythonic-way-to-select-list-elements-with-different-probability
    :param population:
    :param weights:
    :return:

    Args:
        size:
    """
    assert len(population) == len(weights)
    cdf_vals = cdf(weights)
    x = random(size=size)
    if not size:
        x = [x]
    for e in x:
        idx = bisect.bisect(cdf_vals, e)
        yield population[idx]
