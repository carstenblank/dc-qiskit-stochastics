import datetime
import logging

import numpy as np

from dc_qiskit_stochastics.qiskit_util import ibmq_ourense, ibmqx2
from dc_qiskit_stochastics.discrete_stochastic_process import DiscreteStochasticProcess
from dc_qiskit_stochastics.error_mitigation import create_error_mitigation_experiment

import dc_quantum_scheduling as scheduling
from dc_quantum_scheduling.qiskit.qiskit_provider import set_provider_config
from dc_quantum_scheduling import client

from gBm_delta_data import S_0, r, time_to_maturity, mu, sigma


logging.basicConfig(format=logging.BASIC_FORMAT, level='INFO')
[logging.getLogger(name).setLevel('WARN') for name in logging.root.manager.loggerDict if name.startswith('qiskit')]
logging.getLogger(__name__).setLevel('INFO')

LOG = logging.getLogger(__name__)


set_provider_config(hub='ibm-q', group='open', project='main')
client.set_url('http://localhost:8080')


if __name__ == "__main__":

    version = 'v001'

    n = 4

    shots = 8192 * 1

    P = 100
    L = 100

    backend = ibmqx2
    backend_name = "sim" if backend is None else backend().name()

    l_array = np.arange(-L, L + 1)
    fourier_coefficients = np.asarray(
        [-1.0j / (2 * np.pi * l) * np.exp(-2 * np.pi ** 2 * l ** 2 / P ** 2) if l != 0 else 1 / 2
         for l in l_array])
    char_func_input = 2 * np.pi / P * l_array

    strike_prices = np.asarray([25, 55, 85, 105, 110, 115, 120, 125, 130, 160, 190, 220])
    prepared_experiments = []
    for strike_price in strike_prices:
        LOG.info(f'=============== Starting Preparation for STRIKE_PRICE={strike_price} =====================')

        initial_value = (np.log(S_0) - np.log(strike_price) + (r + sigma**2 / 2) * time_to_maturity) \
                        / (sigma * np.sqrt(time_to_maturity))

        x_1 = (mu - sigma**2/2) / (n * sigma * np.sqrt(time_to_maturity)) \
            - 1 / (np.sqrt(n) * np.sqrt(time_to_maturity))
        x_2 = (mu - sigma**2/2) / (n * sigma * np.sqrt(time_to_maturity)) \
            + 1 / (np.sqrt(n) * np.sqrt(time_to_maturity))

        probabilities = np.asarray(n * [[.5, .5]])
        realizations = np.asarray(n * [[x_1, x_2]])

        dsp: DiscreteStochasticProcess = DiscreteStochasticProcess(
            initial_value=initial_value, probabilities=probabilities,
            realizations=realizations
        )

        LOG.info(f'=============== Created DSP for STRIKE_PRICE={strike_price} =====================')

        configuration = {
            'shots': shots,
            'tags': ['gBm_delta_donsker_qc_fourier', f'{datetime.datetime.now().date().isoformat()}',
                     f'version={version}',
                     f'P={P}', f'L={L}', f'shots={shots}', f'K={strike_price}',
                     f'trans_backend={backend_name}'],
            'no_noise': False,
            'strike_price': strike_price
        }

        exp: scheduling.PreparedExperiment = dsp.characteristic_function(
            evaluations=char_func_input,
            other_arguments=configuration,
            transpiler_target_backend=backend
        )
        LOG.info(f'=============== Preparation of Main Experiment DONE for STRIKE_PRICE={strike_price} =====================')
        prepared_mitigation = create_error_mitigation_experiment(exp)
        LOG.info(f'=============== Preparation of Mitigation Experiment DONE for STRIKE_PRICE={strike_price} =====================')

        LOG.info(f'=============== Saving Experiments for STRIKE_PRICE={strike_price} =====================')
        exp.tags += ['main experiment']

        key_experiment = client.save_prepared_experiment(exp)
        key_mitigation = client.save_prepared_experiment(prepared_mitigation)

        prepared_experiments.append((key_experiment, key_mitigation))

        LOG.info(f'======> STRIKE_PRICE={strike_price} / Experiment key={key_experiment}')
        LOG.info(f'======> STRIKE_PRICE={strike_price} / Mitigation key={key_mitigation}')
        LOG.info(f'=============== Finished Preparation for STRIKE_PRICE={strike_price} =====================')

    # for strike_price, (key_experiment, key_mitigation) in zip(strike_prices, prepared_experiments):
    #     LOG.info(f'======> STRIKE_PRICE={strike_price} / Experiment key={key_experiment}')
    #     LOG.info(f'======> STRIKE_PRICE={strike_price} / Mitigation key={key_mitigation}')
    #     client.run_experiment(key_experiment, backend=backend())
    #     client.run_experiment(key_mitigation, backend=backend())
