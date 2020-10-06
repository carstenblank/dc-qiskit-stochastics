import logging
import re
import unittest
from typing import List, Optional

import numpy as np
import qiskit
import qiskit.result
from ddt import ddt, data as test_data, unpack
from nptyping import NDArray
from qiskit.providers.aer.backends.aerbackend import AerBackend

import dsp_data
import dc_qc_random_walks.benchmark as benchmark
from dc_qc_random_walks.qiskit.dsp_common import apply_initial, x_measurement, y_measurement
from dc_qc_random_walks.qiskit.dsp_independent import index_independent_prep

logging.basicConfig(format=logging.BASIC_FORMAT, level='DEBUG')
LOG = logging.getLogger(__name__)


@ddt
class QiskitDspCircuitTests(unittest.TestCase):
    @staticmethod
    def report_progress(job_id: str, job_status: qiskit.providers.JobStatus, job: qiskit.providers.BaseJob):
        LOG.info(f'Processing {job_id} with status {job_status}...')

    def assert_statevector(self, probabilities: np.ndarray, realizations: np.ndarray,
                           scaling: float, initial_value: float,job: qiskit.providers.BaseJob,
                           measurement: Optional[str] = None):
        LOG.info(f'Asserting {job.job_id()} with status {job.status()}...')
        if job.status() == qiskit.providers.JobStatus.DONE:
            result: qiskit.result.Result = job.result()
            vector = result.get_statevector()

            # Gather facts from the probabilities about the number k (realizations)
            # and the number of qubits to encode a this number
            k = len(probabilities[0, :])
            index_register_size = int(np.ceil(np.log2(k)))

            # The bits show the string length of bits to encode the Hilbert space of the state vector (dimension of it)
            bits = int(np.log2(len(vector)))

            # First assertion: the bits to encode the dimension of the state vector must be
            # divisible by the index register size if the data bit is removed
            self.assertTrue((bits - 1) % index_register_size == 0, msg='Bit length of state vector must be 1 data bit + l level index.')

            # If the last assertion passed, the divisor os the number of levels, so it must be equal
            levels = (bits - 1) / index_register_size
            self.assertTrue(levels == len(probabilities), msg='Bit length of state vector must show the correct number of levels.')

            # Save the resulting expected vector to compare later. We do this because we want to
            # get the full information first, before we fail the test
            expected_vector: List[complex] = []
            phase_diff_list: List[float] = []

            for index, value in enumerate(vector):
                basis_state_bits = "{0:b}".format(index).zfill(bits)  # [::-1]
                data_state = int(basis_state_bits[-1], 2)
                values: List[str] = re.findall(''.join(index_register_size*['.']), basis_state_bits[:-1])
                values: List[int] = [int(v, 2) for v in values]
                LOG.debug(f'Found the extraction indices to be {values}.')

                all_probs = probabilities[np.arange(len(probabilities)), values]
                LOG.debug(f'Found the extraction probabilities to be {all_probs}.')
                all_reals = realizations[np.arange(len(realizations)), values]
                LOG.debug(f'Found the extraction realizations to be {all_reals}.')

                p = 1 / np.sqrt(2) * np.prod(np.sqrt(all_probs))
                x = np.sum(all_reals)

                if measurement is None:
                    expected_value: complex = p * np.exp(
                        1.0j * scaling * (initial_value + x) if data_state == 1 else 0.0
                    )
                elif measurement == 'sigma_x':
                    expected_value: complex = p * np.exp(
                        (-1.0)**(1 - data_state) * 1.0j * scaling * (initial_value + x)
                    )
                elif measurement == 'sigma_y':
                    expected_value: complex = p * np.exp(
                        (-1.0)**(1 - data_state) * 1.0j * (np.pi/2 + scaling * (initial_value + x))
                    )
                else:
                    self.fail("If a measurement is given, it must either 'sigma_x' or 'sigma_y'.")

                expected_vector.append(expected_value)

                # As quantum mechanics is neglecting a global phase, we need to find the phase difference.
                # If later we find that all phase differences are the same, we have a global phase
                # and need to adjust for it.
                phase_diff = np.angle(expected_value) - np.angle(value)
                phase_diff_list.append(phase_diff if phase_diff > 10e-6 else phase_diff + 2 * np.pi)
                LOG.info(f'Basis: {basis_state_bits} (data/indices: {data_state}/{values}), '
                         f'Expected: {np.abs(expected_value):.4f} * e^{np.angle(expected_value):.4f}, '
                         f'Actual: {np.abs(value):.4f} * e^{np.angle(value):.4f}')

            # First we check by pairwise comparison, if we have phase differences all within a small
            # delta, if so, we have found a global phase difference.
            LOG.info(f"Phase differences: {phase_diff_list}")
            [self.assertAlmostEqual(x, y, delta=10e-3, msg='Phase Differences have to negligible.')
             for i, x in enumerate(phase_diff_list)
             for j, y in enumerate(phase_diff_list) if i != j]

            # Checking the equality of expect vs. actual values with the global phase taken into account.
            LOG.info(f"Checking of state vector is as expected (modulo phase shift of): {phase_diff_list[0]}")
            for expected, actual in zip(expected_vector, vector):
                actual_phase_shifted = actual * np.exp(1.0j * phase_diff_list[0])
                LOG.debug(f'Expected: {np.abs(expected):.4f} * e^{np.angle(expected):.4f}, '
                          f'Actual: {np.abs(actual_phase_shifted):.4f} * e^{np.angle(actual_phase_shifted):.4f}')
                self.assertAlmostEqual(expected.real, actual_phase_shifted.real, delta=10e-3)
                self.assertAlmostEqual(expected.imag, actual_phase_shifted.imag, delta=10e-3)

            return expected_vector
        else:
            self.fail("The job is not done. Cannot assert correctness.")

    @test_data(*dsp_data.testing_data)
    @unpack
    def test_create_circuit(self, scaling: float, initial_value: float, probabilities: NDArray,
                            realizations: NDArray, apply_func):
        LOG.info(f"Data: scaling={scaling}, initial value={initial_value}, "
                 f"probabilities={list(probabilities)}, realizations={list(realizations)},"
                 f"applied function={apply_func.__name__}.")

        qc = qiskit.QuantumCircuit(name='dsp_simulation')

        LOG.info(f"Initializing with {initial_value} and scaling {scaling}.")
        init_qc = apply_initial(initial_value, scaling)
        qc.extend(init_qc)

        for level, (p, r) in enumerate(zip(probabilities, realizations)):
            LOG.info(f"Adding level {level}: {p} with {r} and scaling {scaling}.")
            qc_index = index_independent_prep(level, p)
            qc_level_l = apply_func(level, r, scaling)
            qc.extend(qc_index)
            qc.extend(qc_level_l)

        LOG.info(f"Circuit:\n{qc.draw(output='text', fold=-1)}")

        qc_compiled = qiskit.transpile(qc, optimization_level=3,basis_gates=['id', 'u1', 'u2', 'u3', 'cx'])
        LOG.info(f"Circuit:\n{qc_compiled.draw(output='text', fold=-1)}")

        backend: qiskit.providers.aer.StatevectorSimulator = qiskit.Aer.get_backend('statevector_simulator')

        job: qiskit.providers.aer.AerJob = qiskit.execute(qc_compiled, backend)
        job.wait_for_final_state(callback=QiskitDspCircuitTests.report_progress, wait=1)

        self.assert_statevector(probabilities, realizations, scaling, initial_value, job)

    @test_data(*dsp_data.testing_data)
    @unpack
    def test_calculate_x_measurement_cos(self, scaling: float, initial_value: float, probabilities: NDArray,
                                         realizations: NDArray, apply_func):
        LOG.info(f"Data: scaling={scaling}, initial value={initial_value}, "
                 f"probabilities={list(probabilities)}, realizations={list(realizations)},"
                 f"applied function={apply_func.__name__}.")

        qc = qiskit.QuantumCircuit(name='dsp_simulation')

        LOG.info(f"Initializing with {initial_value} and scaling {scaling}.")
        init_qc = apply_initial(initial_value, scaling)
        qc.extend(init_qc)

        for level, (p, r) in enumerate(zip(probabilities, realizations)):
            LOG.info(f"Adding level {level}: {p} with {r} and scaling {scaling}.")
            qc_index = index_independent_prep(level, p)
            qc_level_l = apply_func(level, r, scaling)
            qc.extend(qc_index)
            qc.extend(qc_level_l)

        qc.extend(x_measurement())

        LOG.info(f"Circuit:\n{qc.draw(output='text', fold=-1)}")

        qc_compiled = qiskit.transpile(qc, optimization_level=3,basis_gates=['id', 'u1', 'u2', 'u3', 'cx'])
        LOG.info(f"Circuit:\n{qc_compiled.draw(output='text', fold=-1)}")

        backend: qiskit.providers.aer.StatevectorSimulator = qiskit.Aer.get_backend('qasm_simulator')

        job: qiskit.providers.aer.AerJob = qiskit.execute(qc_compiled, backend, shots=2**16)
        job.wait_for_final_state(callback=QiskitDspCircuitTests.report_progress, wait=1)

        expected_cos = benchmark.brute_force_rw_ind(probabilities, realizations, initial_value, scaling, np.cos)
        result: qiskit.result.Result = job.result()

        counts = result.get_counts()
        p: float = (counts.get('0', 0) - counts.get('1', 0)) / (counts.get('0', 0) + counts.get('1', 0))

        LOG.info(f"Assertion: expected={expected_cos}, actual={p}, diff={np.abs(expected_cos - p)}")
        self.assertAlmostEqual(expected_cos, p, delta=10e-3)

    @test_data(*dsp_data.testing_data)
    @unpack
    def test_calculate_y_measurement_sin(self, scaling: float, initial_value: float, probabilities: np.ndarray,
                                         realizations: np.ndarray, apply_func):
        LOG.info(f"Data: scaling={scaling}, initial value={initial_value}, "
                 f"probabilities={list(probabilities)}, realizations={list(realizations)},"
                 f"applied function={apply_func.__name__}.")

        qc = qiskit.QuantumCircuit(name='dsp_simulation')

        LOG.info(f"Initializing with {initial_value} and scaling {scaling}.")
        init_qc = apply_initial(initial_value, scaling)
        qc.extend(init_qc)

        for level, (p, r) in enumerate(zip(probabilities, realizations)):
            LOG.info(f"Adding level {level}: {p} with {r} and scaling {scaling}.")
            qc_index = index_independent_prep(level, p)
            qc_level_l = apply_func(level, r, scaling)
            qc.extend(qc_index)
            qc.extend(qc_level_l)

        qc.extend(y_measurement())

        LOG.info(f"Circuit:\n{qc.draw(output='text', fold=-1)}")

        qc_compiled = qiskit.transpile(qc, optimization_level=3,basis_gates=['id', 'u1', 'u2', 'u3', 'cx'])
        LOG.info(f"Circuit:\n{qc_compiled.draw(output='text', fold=-1)}")

        backend: qiskit.providers.aer.StatevectorSimulator = qiskit.Aer.get_backend('qasm_simulator')

        job: qiskit.providers.aer.AerJob = qiskit.execute(qc_compiled, backend, shots=2**16)
        job.wait_for_final_state(callback=QiskitDspCircuitTests.report_progress, wait=1)

        expected_cos = benchmark.brute_force_rw_ind(probabilities, realizations, initial_value, scaling, np.sin)
        result: qiskit.result.Result = job.result()

        counts = result.get_counts()
        p: float = (counts.get('0', 0) - counts.get('1', 0)) / (counts.get('0', 0) + counts.get('1', 0))

        LOG.info(f"Assertion: expected={expected_cos}, actual={p}, diff={np.abs(expected_cos - p)}")
        self.assertAlmostEqual(expected_cos, p, delta=10e-3)
