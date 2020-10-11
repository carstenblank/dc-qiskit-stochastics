import qiskit
from qiskit.providers.aer.backends import aerbackend
from qiskit.providers.ibmq import ibmqbackend

from .qiskit_provider import provider


def ibmq_ourense() -> 'ibmqbackend.IBMQBackend':
    return provider().get_backend('ibmq_ourense')


def ibmq_vigo() -> 'ibmqbackend.IBMQBackend':
    return provider().get_backend('ibmq_vigo')


def qasm_simulator() -> 'aerbackend.AerBackend':
    return qiskit.Aer.get_backend('qasm_simulator')


def ibmq_simulator() -> 'ibmqbackend.IBMQBackend':
    return provider().get_backend('ibmq_qasm_simulator')
