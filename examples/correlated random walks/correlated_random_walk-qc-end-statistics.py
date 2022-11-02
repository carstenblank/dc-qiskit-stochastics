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

import matplotlib.pyplot as plt
import numpy as np

from dc_qiskit_stochastics import benchmark
from dc_quantum_scheduling.qiskit.qiskit_provider import set_provider_config
from dc_quantum_scheduling import client

logging.basicConfig(format=logging.BASIC_FORMAT, level='INFO')
[logging.getLogger(name).setLevel('WARN')
 for name in logging.root.manager.loggerDict
 if name.startswith('qiskit')]
logging.getLogger(__name__).setLevel('INFO')

LOG = logging.getLogger(__name__)
set_provider_config(hub='ibm-q', group='open', project='main')
client.set_url('http://localhost:8080')


if __name__ == "__main__":
    # =======================================================
    # Pull experiments
    # =======================================================
    tags = ['correlated_walk', 'version=v002', 'execution_backend=ibmq_ourense', 'shots=32768']
    experiment_key = client.get_experiments(*tags, 'main experiment')[0]
    mitigation_key = client.get_experiments(*tags, 'error mitigation')[0]

    if client.get_status(experiment_key) != 'finished' or client.get_status(mitigation_key) != 'finished':
        LOG.info("Not Done Yet!")
        exit(0)

    finished_experiment = client.get_experiment(experiment_key)
    finished_mitigation = client.get_experiment(mitigation_key)

    # =======================================================
    # Get the results & process data
    # =======================================================
    char_func = finished_experiment.get_output(finished_mitigation.get_output())
    char_func_no_mitigation = finished_experiment.get_output()

    evaluations = finished_experiment.arguments
    probabilities = finished_experiment.parameters.get('probabilities')
    realizations = finished_experiment.parameters.get('realizations')
    initial_value = finished_experiment.parameters.get('initial_value')

    # =======================================================
    # Get the theoretical values as benchmark
    # =======================================================
    theory_char_func, histogram = benchmark.brute_force_correlated_walk(
        probabilities=probabilities,
        realizations=realizations,
        initial_value=initial_value,
        scaling=evaluations,
        func=lambda x: np.exp(1.0j * x),
        return_hist=True
    )
    plt.bar(x=histogram[:, 0], height=histogram[:, 1], width=1)
    plt.title('Probability density')
    plt.show()

    # =======================================================
    # Then calculate statistics of the process
    # =======================================================
    expectation_value = benchmark.brute_force_correlated_walk(
        probabilities=probabilities,
        realizations=realizations,
        initial_value=initial_value,
        scaling=1,
        func=lambda x: x
    )
    second_moment = benchmark.brute_force_correlated_walk(
        probabilities=probabilities,
        realizations=realizations,
        initial_value=initial_value,
        scaling=1,
        func=lambda x: x**2
    )
    std_deviation = np.sqrt(second_moment - expectation_value**2)

    kurtosis = benchmark.brute_force_correlated_walk(
        probabilities=probabilities,
        realizations=realizations,
        initial_value=initial_value,
        scaling=1,
        func=lambda x: ((x - expectation_value)/second_moment)**4
    )
    LOG.info(f'Data: initial value: {initial_value}, x_1: {list(realizations[:, 0])} and x_2: {list(realizations[:, 1])} '
             f'with persistence h: {list(probabilities[:,0])}, l: {list(probabilities[:,1])}')
    LOG.info(F'Expectation Value: {expectation_value} and Std Deviation: {std_deviation}, kurtosis: {kurtosis}')
