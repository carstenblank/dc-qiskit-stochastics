import logging

import numpy as np
import qiskit
from dc_qiskit_algorithms import UniformRotationGate
from qiskit.circuit import Parameter
from qiskit.circuit.library import U1Gate
from scipy import sparse

LOG = logging.getLogger(__name__)


def apply_initial(value: float, scaling_factor: Parameter) -> qiskit.QuantumCircuit:
    """
    This function initializes the circuit using Proposition 1 of the paper:
    First we need a Hadamard and then we rotate by the value * scaling factor * 2
    with the R_z rotation.
    :param value: The initial value
    :param scaling_factor: The scaling factor to be used when computing the characteristic function
    :return: The initial quantum circuit with the data system only
    """
    qc = qiskit.QuantumCircuit(name='initial_rotation')
    qreg_data = qiskit.QuantumRegister(1, 'data')
    qc.add_register(qreg_data)
    # BY Proposition 1 we need to start in a superposition state
    qc.h(qreg_data)
    # Then the initial rotation
    # qc.append(RZGate(2*scaling_factor * value), qreg_data)
    qc.u1(scaling_factor * value, qreg_data)
    return qc


def apply_level(level: int, realizations: np.array, scaling_factor: Parameter) -> qiskit.QuantumCircuit:
    # FIXME: Tests dont pass!
    k, = realizations.shape
    qubits_k = int(np.ceil(np.log2(k)))
    qc = qiskit.QuantumCircuit(name=f'level_{level}')
    qreg_index = qiskit.QuantumRegister(qubits_k, f'level_{level}')
    qreg_data = qiskit.QuantumRegister(1, 'data')
    qc.add_register(qreg_index)
    qc.add_register(qreg_data)
    alpha = sparse.dok_matrix([realizations]).transpose()
    LOG.debug(f"Will add a uniform rotation gate with u1({scaling_factor} * {realizations})")
    qc.append(UniformRotationGate(lambda theta: U1Gate(scaling_factor * theta), alpha), [qreg_index, qreg_data])
    return qc


def apply_level_two_realizations(level: int, realizations: np.array, scaling_factor: Parameter) -> qiskit.QuantumCircuit:
    k, = realizations.shape
    assert k == 2
    qubits_k = int(np.ceil(np.log2(k)))
    qc = qiskit.QuantumCircuit(name=f'level_{level}')
    qreg_index = qiskit.QuantumRegister(qubits_k, f'level_{level}')
    qreg_data = qiskit.QuantumRegister(1, 'data')
    qc.add_register(qreg_index)
    qc.add_register(qreg_data)
    x_l1 = realizations[0]
    x_l2 = realizations[1]
    qc.cu1(theta=scaling_factor * x_l2, control_qubit=qreg_index, target_qubit=qreg_data)
    qc.x(qreg_index)
    qc.cu1(theta=scaling_factor * x_l1, control_qubit=qreg_index, target_qubit=qreg_data)
    qc.x(qreg_index)
    return qc


def x_measurement() -> qiskit.QuantumCircuit:
    qc = qiskit.QuantumCircuit(name='initial_rotation')
    qreg_data = qiskit.QuantumRegister(1, 'data')
    creg_data = qiskit.ClassicalRegister(1, 'output')
    qc.add_register(qreg_data)
    qc.add_register(creg_data)
    qc.h(qreg_data)
    qc.measure(qreg_data, creg_data)
    return qc


def y_measurement() -> qiskit.QuantumCircuit:
    qc = qiskit.QuantumCircuit(name='initial_rotation')
    qreg_data = qiskit.QuantumRegister(1, 'data')
    creg_data = qiskit.ClassicalRegister(1, 'output')
    qc.add_register(qreg_data)
    qc.add_register(creg_data)
    qc.z(qreg_data)
    qc.s(qreg_data)
    qc.h(qreg_data)
    qc.measure(qreg_data, creg_data)
    return qc
