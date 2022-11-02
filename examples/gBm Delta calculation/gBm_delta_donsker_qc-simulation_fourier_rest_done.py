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
import matplotlib.pyplot as plt

from dc_quantum_scheduling.qiskit.qiskit_provider import set_provider_config
from dc_quantum_scheduling import client, FinishedExperiment, processor
from dc_quantum_scheduling.io import save
from gBm_delta_data import S_0, r, time_of_maturity, mu, sigma, time_evaluation, K

logging.basicConfig(format=logging.BASIC_FORMAT, level='INFO')
logging.getLogger(__name__).setLevel('INFO')

LOG = logging.getLogger(__name__)


set_provider_config(hub='ibm-q', group='open', project='main')
client.set_url('http://localhost:8080')


if __name__ == "__main__":

    experiment_keys = client.get_experiments('gBm_delta_donsker_qc_fourier', 'trans_backend=ibmqx2', 'version=v001')

    prepared_experiments: dict = {}
    for key in experiment_keys:
        status = client.get_status(key)
        LOG.info(f'Loading {key}: {status}')
        assert status == 'finished'
        experiment: FinishedExperiment = client.get_experiment(key, raise_error=False)
        strike_price = float(experiment.find_tag('K=')[0].replace('K=', ''))
        if strike_price not in prepared_experiments:
            prepared_experiments[strike_price] = [None, None]
        if 'main experiment' in experiment.tags:
            prepared_experiments[strike_price][0] = experiment
            LOG.info(f'Main Experiment {key} for {strike_price}: {experiment}')
        elif 'error mitigation' in experiment.tags:
            prepared_experiments[strike_price][1] = experiment
            LOG.info(f'Mitigation Experiment {key} for {strike_price}: {experiment}')

    exp_vals = []
    sim_exp_vals = []
    experiment: FinishedExperiment
    mitigation: FinishedExperiment
    P_list = []
    L_list = []
    backend_name_list = []
    execution_backend_name_list = []
    shots_list = []
    for strike_price, [experiment, mitigation] in prepared_experiments.items():
        LOG.info(f'======> STRIKE_PRICE={strike_price} / Experiment={experiment}')
        LOG.info(f'======> STRIKE_PRICE={strike_price} / Mitigation={mitigation}')

        # ====== Simulations =======
        # Error mitigation
        sim_running_mitigation = processor.execute_simulation(mitigation)
        LOG.info(f'Mitigation running for {strike_price}')
        sim_finished_mitigation = sim_running_mitigation.wait()
        LOG.info(f'Mitigation done for {strike_price}')

        # Run the experiment
        sim_running_experiment = processor.execute_simulation(experiment)
        LOG.info(f'Experiment running for {strike_price}')
        sim_finished_experiment = sim_running_experiment.wait()
        LOG.info(f'Experiment done for {strike_price}')

        output = experiment.get_output(mitigation.get_output())
        char_func_output = np.asarray(output)

        sim_output = sim_finished_experiment.get_output(sim_finished_mitigation.get_output())
        sim_char_func_output = np.asarray(sim_output)

        P = int(experiment.get_tag_param('P'))
        L = int(experiment.get_tag_param('P'))
        backend_name = experiment.get_tag_param('trans_backend')
        shots = int(experiment.get_tag_param('shots'))
        execution_backend_name_list.append(experiment.execution_backend_name)

        P_list.append(P)
        L_list.append(L)
        backend_name_list.append(backend_name)
        shots_list.append(shots)

        l_array = np.arange(-L, L + 1)
        fourier_coefficients = np.asarray(
            [-1.0j / (2 * np.pi * l) * np.exp(-2 * np.pi ** 2 * l ** 2 / P ** 2) if l != 0 else 1 / 2
             for l in l_array])
        exp_val = char_func_output.dot(fourier_coefficients)
        sim_exp_val = sim_char_func_output.dot(fourier_coefficients)

        LOG.info(f'For strike price {strike_price} the evaluation of the expectation value is {exp_val}.')

        exp_vals.append(exp_val)
        sim_exp_vals.append(sim_exp_val)
        save('../../data/experiments/', simulation=sim_finished_experiment, experiment=experiment)
        save('../../data/experiments/', simulation=sim_finished_mitigation, experiment=mitigation)

    P = set(P_list)
    L = set(L_list)
    backend_name = set(backend_name_list)
    shots = set(shots_list)
    execution_backend_name = set(execution_backend_name_list)
    assert len(P) == 1
    assert len(L) == 1
    assert len(backend_name) == 1
    assert len(shots) == 1
    assert len(execution_backend_name) == 1

    P = P.pop()
    L = L.pop()
    backend_name = backend_name.pop()
    shots = shots.pop()
    execution_backend_name = execution_backend_name.pop()

    n = 4

    expectation_approx = np.asarray(exp_vals)
    sim_expectation_approx = np.asarray(sim_exp_vals)
    strike_prices = list(prepared_experiments.keys())
    plt.scatter(x=strike_prices, y=np.real(expectation_approx))
    plt.scatter(x=strike_prices, y=np.imag(expectation_approx))
    plt.title(
        f"Experiment on {execution_backend_name}\n"
        f'gBm: $S_0$={S_0}/K={K}/r={r}/$\mu$={mu}/$\sigma$={sigma}/$t$={time_evaluation},/$T$={time_of_maturity}\n'
        f'RW-QC (Donkser), n={n}, P={P}, L={L}\n'
        f'backend: shots={shots}, transpiler={backend_name}'
    )
    plt.show()

    df = pd.DataFrame(data=zip(strike_prices, np.real(expectation_approx), np.imag(expectation_approx),
                               np.real(sim_expectation_approx), np.imag(sim_expectation_approx)),
                      columns=['strike price', 'expectation value (real)', 'expectation value (imag)',
                               'expectation value (sim/real)', 'expectation value (sim/imag)'])
    df.to_csv(f'../../data/gBm-delta-donsker-{n}-fourier-qc-{execution_backend_name}-P={P}-L={L}.csv')
