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
import datetime
import logging

import numpy as np

from dc_qiskit_stochastics.qiskit_util import ibmq_ourense, ibmq_simulator, ibmqx2
from dc_quantum_scheduling.qiskit.qiskit_provider import set_provider_config
from dc_qiskit_stochastics.dsp_correlated_walk import CorrelatedWalk
from dc_qiskit_stochastics.error_mitigation import create_error_mitigation_experiment
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
    # Global Settings
    # =======================================================
    version = 'v004'

    # =======================================================
    # Stochastic Process Settings
    # =======================================================
    n = 4
    initial_value = 0

    x_1 = -1
    x_2 = +1

    # These probabilities say that (apart from the first one) how high the probability is to repeat the last realization
    # probabilities = np.asarray([[0.5, 0.5]] + (n - 1) * [[0.9, 0.9]])
    # probabilities = np.asarray([[0.5, 0.5]] + (n - 1) * [[0.5, 0.5]])
    probabilities = np.stack(
        # (np.linspace(0.5, 0.5, n), np.linspace(0.5, 0.5, n)), axis=-1
        (np.linspace(0.5, 1.0, n), np.linspace(0.5, 0.0, n)), axis=-1
    )
    realizations = np.asarray(n * [[x_1, x_2]])

    # =======================================================
    # QC Settings
    # =======================================================
    shots = 4 * 8192
    transpiler_backend = ibmqx2
    execution_backend = ibmqx2
    backend_name = transpiler_backend().name()

    # =======================================================
    # Discretization settings
    # =======================================================
    P = 100
    L = 100

    # =======================================================
    # Second calculate evaluations and create CRW
    # =======================================================
    l_array = np.arange(-L, L + 1)
    evaluations = 2 * np.pi / P * l_array

    correlated_walk = CorrelatedWalk(
        initial_value=initial_value,
        realizations=realizations,
        probabilities=probabilities
    )

    # =======================================================
    # Third create experiments
    # =======================================================
    configuration = {
        'shots': shots,
        'tags': ['correlated_walk',
                 f'{datetime.datetime.now().date().isoformat()}', f'version={version}',
                 f'shots={shots}', f'transpiler_backend={backend_name}',
                 f'execution_backend={execution_backend().name()}', f'P={P}', f'L={L}'],
        'no_noise': False
    }

    experiment = correlated_walk.characteristic_function(
        evaluations=evaluations,
        other_arguments=configuration,
        transpiler_target_backend=transpiler_backend
    )
    experiment_mitigation = create_error_mitigation_experiment(experiment)
    LOG.info(f'=============== Preparation of Mitigation Experiment DONE =====================')

    LOG.info(f'=============== Saving Experiments =====================')
    experiment.tags += ['main experiment']

    key_experiment, key_mitigation = client.execute_experiment(
        main_experiment=experiment,
        mitigation_experiment=experiment_mitigation,
        execution_backend=execution_backend()
    )

    LOG.info(f'======> Experiment key={key_experiment}')
    LOG.info(f'======> Mitigation key={key_mitigation}')
    LOG.info(f'=============== Finished Preparation =====================')
