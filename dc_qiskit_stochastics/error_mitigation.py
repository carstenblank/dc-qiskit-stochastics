import logging
import sys
from typing import List, Optional

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.ignis.mitigation import complete_meas_cal, CompleteMeasFitter
from qiskit.providers.models import QasmBackendConfiguration
from qiskit.qobj import Qobj
from qiskit.transpiler import PassManager

from dc_quantum_scheduling import PreparedExperiment, FinishedExperiment

from . import qobj_mapping, get_default_pass_manager

LOG = logging.getLogger(__name__)


def create_qobj(circuits: List[QuantumCircuit], pass_manager: PassManager, qobj_id, other_arguments, transpiler_target_backend):
    circuits_assembled = [pass_manager.run(qc) for qc in circuits]

    config: QasmBackendConfiguration = transpiler_target_backend.configuration()
    max_shots = config.max_shots
    max_experiments = config.max_experiments if hasattr(config, 'max_experiments') else sys.maxsize
    shots = other_arguments.get('shots', max_shots)
    other_arguments['shots'] = shots

    mapping_matrix: np.array = qobj_mapping(shots, max_shots, max_experiments, len(circuits_assembled))
    shots_per_experiment = int(shots / mapping_matrix.shape[1])
    LOG.debug(mapping_matrix)
    number_of_qobj: int = np.max(mapping_matrix) + 1

    LOG.info(f'For #{len(circuits_assembled)} evaluations each {shots} on the device {transpiler_target_backend.name()} '
             f'with max shots {max_shots} and max no. of experiments per Qobj {max_experiments} '
             f'there are #{number_of_qobj} of Qobj needed. Assembling the circuits now.')

    qobj_list: List[Qobj] = []
    circuit_list: List[List[QuantumCircuit]] = [[] for i in range(number_of_qobj)]

    for row_no, qc in enumerate(circuits_assembled):
        indices = mapping_matrix[row_no]
        for i, qobj_id in enumerate(indices):
            qobj_circuits = circuit_list[qobj_id]
            qobj_circuits.append(qc)

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


def interpret_output(finished_experiment: FinishedExperiment,
                      meas_fitter: Optional[CompleteMeasFitter] = None) -> CompleteMeasFitter:

    if meas_fitter is not None:
        return meas_fitter

    qubit_list = finished_experiment.parameters['qubit_list']
    state_labels = finished_experiment.parameters['state_labels']
    # Get results
    cal_results = []
    for qobj_index in range(len(finished_experiment.qobj_list)):
        r = finished_experiment.to_result(qobj_index)
        cal_results.append(r)

    # Calculate the calibration matrix with the noise model
    return CompleteMeasFitter(cal_results, state_labels, qubit_list=qubit_list, circlabel='mcal')


def create_error_mitigation_experiment(exp: PreparedExperiment) -> PreparedExperiment:
    # def calculate_error_mitigation_matrix(self, backend: Optional[Union[IBMQBackend, AerBackend]] = None):
    # Get the measured qubits
    measured_qubits = set([e.instructions[-1].qubits[0] for q in exp.qobj_list for e in q.experiments])
    LOG.info(f'Qubits measured are {measured_qubits}')

    # Generate the calibration circuits
    qr = qiskit.QuantumRegister(exp.transpiler_backend.configuration().n_qubits)
    qubit_list = list(measured_qubits)
    meas_calibs, state_labels = complete_meas_cal(qubit_list=qubit_list, qr=qr, circlabel='mcal')

    _, pm = get_default_pass_manager(exp.transpiler_backend, exp.parameters)

    external_id = f'{exp.external_id}-error_mitigation'
    qobj_list = create_qobj(meas_calibs,
                            pass_manager=pm,
                            qobj_id=external_id,
                            other_arguments=exp.parameters,
                            transpiler_target_backend=exp.transpiler_backend
                            )

    return PreparedExperiment(
        external_id=f'{exp.external_id}-error_mitigation',
        tags=exp.tags + ['error mitigation'],
        qobj_list=qobj_list,
        arguments=[],
        transpiler_backend=exp.transpiler_backend,
        parameters={
            'qubit_list': qubit_list,
            'state_labels': state_labels,
            **exp.parameters,
        },
        callback=interpret_output
    )