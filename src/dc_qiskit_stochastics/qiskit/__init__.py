import qiskit
from qiskit.providers.aer.backends.aerbackend import AerBackend
from qiskit.providers.ibmq import IBMQBackend

from .qiskit_provider import provider


def ibmq_ourense() -> IBMQBackend:
    return provider().get_backend('ibmq_ourense')  # type: IBMQBackend


def ibmq_vigo() -> IBMQBackend:
    return provider().get_backend('ibmq_vigo')  # type: IBMQBackend


def qasm_simulator() -> AerBackend:
    return qiskit.Aer.get_backend('qasm_simulator')  # type: AerBackend


def ibmq_simulator() -> IBMQBackend:
    return provider().get_backend('ibmq_qasm_simulator')  # type: IBMQBackend
