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
import pandas as pd

from dc_qiskit_stochastics import benchmark, plotting
from dc_quantum_scheduling.qiskit.qiskit_provider import set_provider_config
from dc_quantum_scheduling import client, processor
from dc_quantum_scheduling.io import save

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
    tags = ['correlated_walk', 'version=v004', 'execution_backend=ibmqx2', 'shots=32768']
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
    # Get the simulated values as benchmark
    # =======================================================
    simulated_experiment = processor.execute_simulation(finished_experiment.get_prepared_experiment()).wait()
    simulated_mitigation = processor.execute_simulation(finished_mitigation.get_prepared_experiment()).wait()
    simulated_char_func = simulated_experiment.get_output(simulated_mitigation.get_output())

    # =======================================================
    # Get the theoretical values as benchmark
    # =======================================================
    theory_char_func = benchmark.brute_force_correlated_walk(
        probabilities=probabilities,
        realizations=realizations,
        initial_value=initial_value,
        scaling=evaluations,
        func=lambda x: np.exp(1.0j * x),
        return_hist=False
    )

    # =======================================================
    # Plot all
    # =======================================================
    fig = plotting.plot_characteristic_function(
        simulation=simulated_char_func,
        experiment=char_func,
        theory=theory_char_func)
    fig.show()

    df = pd.DataFrame(
        data=zip(evaluations,
            np.real(char_func), np.imag(char_func),
            np.real(char_func_no_mitigation), np.imag(char_func_no_mitigation),
            np.real(simulated_char_func), np.imag(simulated_char_func),
            np.real(theory_char_func), np.imag(theory_char_func)
        ),
        columns=[
            'evaluations',
            'expectation value (real w/ mitigation)', 'expectation value (imag w/ mitigation)',
            'expectation value (real)', 'expectation value (imag)',
            'expectation value (sim/real)', 'expectation value (sim/imag)',
            'expectation value (theory/real)', 'expectation value (theory/imag)'
        ]
    )
    P = finished_experiment.get_tag_param('P') or '100'
    L = finished_experiment.get_tag_param('L') or '100'
    n = len(finished_experiment.parameters.get('probabilities'))
    shots = finished_experiment.parameters.get('shots')
    version = finished_experiment.get_tag_param('version')
    execution_backend_name = finished_experiment.execution_backend.name()

    df.to_csv(f'../../data/crw-{n}-fourier-qc-{execution_backend_name}-P={P}-L={L}-{version}-shots={shots}.csv')

    save(
        directory=f'../../data/experiments',
        experiment=finished_experiment,
        simulation=simulated_experiment
    )

    save(
        directory=f'../../data/experiments',
        experiment=finished_mitigation,
        simulation=simulated_mitigation
    )
