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
import unittest
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import qiskit
from qiskit.circuit import Parameter

from dc_qiskit_stochastics.benchmark import char_func_asian_option, char_func_asian_option_sm
from dc_qiskit_stochastics.dsp_state_machine import StateMachineDSP
from dc_qiskit_stochastics.simulation.asian_option import AsianOptionPricing, StateMachineDescription
from dc_quantum_scheduling import processor
from dc_quantum_scheduling.models import PreparedExperiment, RunningExperiment, FinishedExperiment
from dsp_data import testing_data_state_machine

logging.basicConfig(format=f'%(asctime)s::{logging.BASIC_FORMAT}', level='ERROR')
LOG = logging.getLogger(__name__)


class StateMachineTest(unittest.TestCase):

    def test_init_success(self):
        data = {
            'initial_value': 0,
            'probabilities': np.asarray([
                [
                    [0.1, 0.6]
                ],
                [
                    [0.1, 0.6, 0.3],
                    [0.5, 0.3, 0.1]
                ],
                [
                    [0.1, 0.6],
                    [0.5, 0.3],
                    [0.1, 0.1]
                ]
            ]),
            'realizations': np.asarray([
                [
                    [1, 1]
                ],
                [
                    [1, 1, 1],
                    [1, 1, 1]
                ],
                [
                    [1, 1],
                    [1, 1],
                    [1, 1]
                ]
            ])
        }

        try:
            StateMachineDSP(data['initial_value'], data['probabilities'], data['realizations'])
        except AssertionError:
            self.fail()

    def test_init_fail(self):
        data = {
            'initial_value': 0,
            'probabilities': np.asarray([
                [
                    [0.1, 0.6]
                ],
                [
                    [0.1, 0.6, 0.3],
                    [0.5, 0.3, 0.1],
                    [0.5, 0.3, 0.1]
                ],
                [
                    [0.1, 0.6],
                    [0.5, 0.3],
                    [0.1, 0.1]
                ]
            ]),
            'realizations': np.asarray([
                [
                    [1, 1]
                ],
                [
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]
                ],
                [
                    [1, 1],
                    [1, 1],
                    [1, 1]
                ]
            ])
        }

        try:
            StateMachineDSP(data['initial_value'], data['probabilities'], data['realizations'])
        except AssertionError:
            return
        self.fail()

    def test_full(self):
        for i, data in enumerate(testing_data_state_machine):
            print(f'Test on data {i}.')

            process = StateMachineDSP(data['initial_value'], data['probabilities'], data['realizations'])
            qc = process._proposition_one_circuit(Parameter('v'))

            qc_t = qiskit.transpile(
                qc, basis_gates=['uni_rot_rx', 'uni_rot_ry', 'uni_rot_rz', 'uni_rot_rx_dg', 'uni_rot_ry_dg',
                                 'uni_rot_rz_dg', 'uni_rot_u1', 'uni_rot_u1_dg', 'h', 'u1']
            )
            print(qc_t.draw(fold=-1))

            qc_t = qiskit.transpile(qc, basis_gates=['u1', 'u2', 'u3', 'cx'], optimization_level=3)
            print(qc_t.draw(fold=-1))
            print(f'Depth of final circuit: {qc_t.depth()}')

    def test_log_normal(self):
        log_normal_data = {
            'risk_free_interest': 0.02,
            'volatility': 0.05,
            'start_value': 80,
            'time_steps': np.asarray([5]),
            'discretization': 2**4
        }

        s0 = log_normal_data['start_value']
        sigma = log_normal_data['volatility']
        mu = log_normal_data['risk_free_interest']
        time_steps = log_normal_data['time_steps']
        discretization = log_normal_data['discretization']

        asian_option_model = AsianOptionPricing(s0, sigma, mu, time_steps, discretization)
        data = asian_option_model.get_state_machine_model()
        v = np.linspace(-0.3, 0.3, num=400)

        # TODO: test the benchmark separately with this:
        # # As this is just the lognormal distribution, we can directly brute-force compute the outcome.
        # phi_v_sim = []
        # for entry in v:
        #     summands = data.probabilities[0] * np.exp(1.0j * entry * data.realizations)
        #     summed = np.sum(summands)
        #     phi_v_sim.append(summed)
        # phi_v_sim = np.asarray(phi_v_sim)
        phi_v_sim = char_func_asian_option_sm(v, asian_option_model)

        # The quantum approach is created here
        state_machine = StateMachineDSP(data.initial_value, data.probabilities, data.realizations)
        pre_exp: PreparedExperiment = state_machine.characteristic_function(
            evaluations=v, other_arguments={'shots': 2000, 'with_barrier': True}
        )
        print(pre_exp.parameters['qc_cos'].draw(fold=-1))
        run_exp: RunningExperiment = processor.execute_simulation(pre_exp)
        fin_exp: Optional[FinishedExperiment] = run_exp.wait()
        phi_v_qc = fin_exp.get_output()

        # The MC benchmark computation
        phi_v = char_func_asian_option(
            asian_option_model.mu,
            asian_option_model.sigma,
            asian_option_model.s0,
            asian_option_model.time_steps,
            samples=2000,
            evaluations=v
        )

        # TODO: remove if not needed
        # s = sigma * np.sqrt(delta_t)
        # mu_tilde = (mu - 0.5 * sigma ** 2) * delta_t + np.log(s0)
        # phi_v_benchmark_m = np.asarray(
        #     [(1.0j * v) ** n / factorial(n) * np.exp(n * mu_tilde + 0.5 * n**2 * s**2) for n in range(1000)]
        # )
        # phi_v = np.sum(phi_v_benchmark_m, axis=0, where=~np.isnan(phi_v_benchmark_m))

        # Plotting party
        plt.plot(v, np.real(phi_v_sim), color='gray', alpha=0.7, label='QC-TH')
        plt.scatter(x=v, y=np.real(phi_v_qc), color='blue', alpha=1.0, label='QC', marker='.')
        plt.plot(v, np.real(phi_v), color='black', label='MC')

        plt.axvline(x=0, color='purple')
        plt.title(f'StateMachineTest.test_log_normal\nReal part (cosine) at {time_steps}')
        plt.ylim((-1.1, 1.1))
        plt.legend()
        plt.show()

        plt.plot(v, np.imag(phi_v_sim), color='gray', alpha=0.7, label='QC-TH')
        plt.scatter(x=v, y=np.imag(phi_v_qc), color='blue', alpha=0.7, label='QC', marker='.')
        plt.plot(v, np.imag(phi_v), color='black', label='MC')

        plt.axvline(x=0, color='purple')
        plt.title(f'StateMachineTest.test_log_normal\nImaginary part (sine) at {time_steps}')
        plt.ylim((-1.1, 1.1))
        plt.legend()
        plt.show()

    def test_asian_option(self):
        log_normal_data = {
            'risk_free_interest': 0.02,
            'volatility': 0.05,
            'start_value': 80,
            'time_steps': np.asarray([5, 10]),
            'discretization': 2**4
        }

        s0 = log_normal_data['start_value']
        sigma = log_normal_data['volatility']
        mu = log_normal_data['risk_free_interest']
        time_steps = log_normal_data['time_steps']
        discretization = log_normal_data['discretization']

        asian_option_model = AsianOptionPricing(s0, sigma, mu, time_steps, discretization)
        data: StateMachineDescription = asian_option_model.get_state_machine_model()
        v = np.linspace(-0.3, 0.3, num=400)

        # The brute force calculation using the state machine description
        phi_v_sim = char_func_asian_option_sm(v, asian_option_model)

        # The quantum approach is created here
        state_machine = StateMachineDSP(data.initial_value, data.probabilities, data.realizations)
        pre_exp: PreparedExperiment = state_machine.characteristic_function(
            evaluations=v, other_arguments={'shots': 2000, 'with_barrier': True}
        )
        print(qiskit.transpile(pre_exp.parameters['qc_cos']).draw(fold=-1))
        run_exp: RunningExperiment = processor.execute_simulation(pre_exp)
        fin_exp: Optional[FinishedExperiment] = run_exp.wait()
        phi_v_qc = fin_exp.get_output()

        # The MC benchmark computation
        phi_v = char_func_asian_option(
            asian_option_model.mu,
            asian_option_model.sigma,
            asian_option_model.s0,
            asian_option_model.time_steps,
            samples=2000,
            evaluations=v
        )

        # Plotting party
        plt.plot(v, np.real(phi_v_sim), color='gray', alpha=0.7, label='QC-TH')
        plt.scatter(x=v, y=np.real(phi_v_qc), color='blue', alpha=1.0, label='QC', marker='.')
        plt.plot(v, np.real(phi_v), color='black', label='MC')
        plt.axvline(x=0, color='purple')
        plt.title(f'StateMachineTest.test_asian_option\nReal part (cosine) at {time_steps}')
        plt.ylim((-1.1, 1.1))
        plt.legend()
        plt.show()

        plt.plot(v, np.imag(phi_v_sim), color='gray', alpha=0.7, label='QC-TH')
        plt.scatter(x=v, y=np.imag(phi_v_qc), color='blue', alpha=0.7, label='QC', marker='.')
        plt.plot(v, np.imag(phi_v), color='black', label='MC')
        plt.axvline(x=0, color='purple')
        plt.title(f'StateMachineTest.test_asian_option\nImaginary part (sine) at {time_steps}')
        plt.ylim((-1.1, 1.1))
        plt.legend()
        plt.show()
