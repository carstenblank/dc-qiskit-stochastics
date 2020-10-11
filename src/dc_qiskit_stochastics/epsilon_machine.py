import logging
from typing import Optional, Dict, Any, Union

import numpy as np
import qiskit
from dc_qiskit_algorithms import UniformRotationGate, QuantumFourierTransformGate
from dc_qiskit_algorithms.MöttönenStatePreparation import get_alpha_y
from dc_qiskit_algorithms.Qft import get_theta
from hmmlearn import hmm
from hmmlearn.hmm import MultinomialHMM
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import Gate, Parameter
from qiskit.circuit.library import RYGate, CU1Gate
from scipy import sparse

LOG = logging.basicConfig(format=logging.BASIC_FORMAT)


class DraperAdder(Gate):

    num_qubits_a: int
    num_qubits_b: int

    def __init__(self, num_qubits_a: int, num_qubits_b: int, label: Optional[str] = None) -> None:
        super().__init__("draper", num_qubits_a + num_qubits_b, [], label)
        self.num_qubits_a = num_qubits_a
        self.num_qubits_b = num_qubits_b

    def _define(self):
        a = QuantumRegister(self.num_qubits_a, "a")
        b = QuantumRegister(self.num_qubits_b, "b")
        qc = QuantumCircuit(a, b, name=self.name)

        qc.append(QuantumFourierTransformGate(len(a)), a, [])

        for b_index in reversed(range(len(b))):
            theta_index = 1
            for a_index in reversed(range(b_index + 1)):
                qc.append(CU1Gate(get_theta(theta_index)), [b[b_index], a[a_index]], [])
                theta_index += 1

        qc.append(QuantumFourierTransformGate(len(a)).inverse(), a, [])

        self.definition = qc


class RYGateSpecial(object):

    scaling: Parameter

    def __init__(self, scaling: Parameter):
        self.scaling = scaling

    def gate(self):
        return lambda phi: RYGate(self.scaling * (-phi))


# class MultinomialHmmSpecial(MultinomialHMM):
#
#     def __init__(self, n_components=1, startprob_prior=1.0, transmat_prior=1.0, algorithm="viterbi", random_state=None,
#                  n_iter=10, tol=1e-2, verbose=False, params="ste", init_params="ste"):
#         super().__init__(n_components, startprob_prior, transmat_prior, algorithm, random_state, n_iter, tol, verbose,
#                          params, init_params)
#
#     def emitted_symbol(self, state):
#         return self.emissionprob_[state, :].argmax()
#
#     def _generate_sample_from_state(self, state, random_state=None):
#         most_likely_emmission = self.emitted_symbol(state)
#         return [most_likely_emmission]


class HiddenMarkovModelAlternative(object):

    emitted_symbols_map: Dict[int, float]
    accumulator_register: QuantumRegister
    state_register: QuantumRegister
    number_of_qubits: int
    number_of_states: int
    model: hmm.MultinomialHMM

    def __init__(self, model: hmm.MultinomialHMM, emitted_symbols_map: Optional[Dict[int, float]] = None) -> None:
        self.emitted_symbols_map = emitted_symbols_map or {}
        self.model = model
        self.number_of_states = self.model.n_components * self.model.n_features
        self.number_of_qubits = int(np.ceil(np.log2(self.number_of_states)))

        self.state_register = QuantumRegister(size=self.number_of_qubits, name='state')
        self.accumulator_register = QuantumRegister(size=1, name='accumulator')

    def _create_index(self, level: int, scaling: Parameter):

        angles = {}
        for k in range(1, self.number_of_qubits + 1):
            size_of_vector = 2 ** (2 * self.number_of_qubits - k)
            vector = sparse.dok_matrix((size_of_vector, 1))
            angles[k] = vector

        transition_matrix: np.ndarray = self.model.transmat_

        for state in range(self.number_of_states):
            transitions: np.ndarray = transition_matrix[state]
            a = sparse.dok_matrix([transitions]).transpose()
            for k in range(1, self.number_of_qubits + 1):
                resulting_angles = angles[k]
                alpha_vector = get_alpha_y(a, self.number_of_qubits, k)
                offset = state * 2 ** (self.number_of_qubits - k)
                for key, alpha in alpha_vector.items():
                    resulting_angles[offset + key[0], key[1]] = alpha

        level_register = QuantumRegister(size=self.number_of_qubits, name=f'level_{level}')
        qc: QuantumCircuit = QuantumCircuit(self.state_register, level_register, name=f'circuit_level_{level}')
        for k in range(1, self.number_of_qubits + 1):
            alpha_angles = angles[k]
            control = list(self.state_register) + list(level_register)[0:self.number_of_qubits - k]
            target = level_register[self.number_of_qubits - k]
            # FIXME: scaling must be adjusted too!
            qc.append(UniformRotationGate(gate=RYGateSpecial(scaling).gate(), alpha=alpha_angles), control + [target], [])

        return qc

    def _create_accumulator_circuit(self, level: int, scaling: Parameter):
        angles: sparse.dok_matrix = sparse.dok_matrix((2**(2*self.number_of_qubits), 1))

        for state in range(self.number_of_states):
            for transition in range(self.number_of_states):
                index = state + transition * self.number_of_states
                # The emitted symbol that occurs when going from state to state + transition!
                target_state = (state + transition) % self.model.n_components
                emitted_symbol = 0  # TODO: do
                angles[index, 0] = self.emitted_symbols_map.get(emitted_symbol, emitted_symbol)

        level_register = QuantumRegister(size=self.number_of_qubits, name=f'level_{level}')
        qc: QuantumCircuit = QuantumCircuit(self.state_register, level_register, self.accumulator_register,
                                            name=f'circuit_accumulator_{level}')
        control = list(self.state_register) + list(level_register)
        target = list(self.accumulator_register)[0]
        # FIXME: scaling must be adjusted too!
        qc.append(
            UniformRotationGate(gate=RYGateSpecial(scaling).gate(), alpha=angles), control + [target], []
        )

        return qc

    def _create_adder(self, level: int):
        level_register = QuantumRegister(size=self.number_of_qubits, name=f'level_{level}')
        qc: QuantumCircuit = QuantumCircuit(self.state_register, level_register,
                                            name=f'circuit_adder_{level}')
        control = list(self.state_register) + list(level_register)
        if self.number_of_qubits == 1:
            qc.cx(level_register, self.state_register)
        else:
            qc.append(
                DraperAdder(num_qubits_a=self.number_of_qubits, num_qubits_b=self.number_of_qubits), control, []
            )
        return qc


if __name__ == "__main__":
    samples = 10*[[0], [1], [1], [1], [1], [2], [2], [0], [1], [1], [1], [0], [2]]
    model = hmm.MultinomialHMM(n_components=4)
    model.fit(samples)

    samples = model.sample(10)

    q_model = HiddenMarkovModel(model)

    qc: QuantumCircuit = QuantumCircuit(name='hmm_circuit')
    scaling_v: Parameter = Parameter('v')
    steps = 3
    for level in range(1, steps + 1):
        level_qc = q_model._create_index(level, scaling_v)
        qc.extend(level_qc)
        # qc.barrier()

        acc_qc = q_model._create_accumulator_circuit(level, scaling_v)
        qc.extend(acc_qc)
        # qc.barrier()

        update_state_qc = q_model._create_adder(level)
        qc.extend(update_state_qc)

        qc.barrier()

    print(qc.draw(fold=-1))

    transpiled_qc = qiskit.transpile(
        qc,
        optimization_level=3,
        basis_gates=[
            'u1', 'u2', 'u3', 'cx', 'id',
            # 'cu1', 'h'
        ]
    )
    print(transpiled_qc.draw(fold=-1))
    print(transpiled_qc.depth(), transpiled_qc.width())
