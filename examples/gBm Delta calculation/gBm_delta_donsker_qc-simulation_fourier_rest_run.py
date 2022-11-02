import logging

from dc_quantum_scheduling import client, PreparedExperiment
from dc_quantum_scheduling.qiskit.qiskit_provider import set_provider_config

from dc_qiskit_stochastics.qiskit_util import ibmqx2

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

        if status == 'error':
            error = client.get_error(key)
            LOG.warning(error)
            client.clear_error(key)
        elif status != 'prepared':
            continue

        experiment: PreparedExperiment = client.get_experiment(key, raise_error=True)
        strike_price = float(experiment.find_tag('K=')[0].replace('K=', ''))
        if strike_price not in prepared_experiments:
            prepared_experiments[strike_price] = [None, None]
        if 'main experiment' in experiment.tags:
            prepared_experiments[strike_price][0] = key
            LOG.info(f'Main Experiment {key} for {strike_price}: {experiment}')
        elif 'error mitigation' in experiment.tags:
            prepared_experiments[strike_price][1] = key
            LOG.info(f'Mitigation Experiment {key} for {strike_price}: {experiment}')

    for strike_price, (key_experiment, key_mitigation) in prepared_experiments.items():
        LOG.info(f'======> STRIKE_PRICE={strike_price} / Experiment key={key_experiment}')
        LOG.info(f'======> STRIKE_PRICE={strike_price} / Mitigation key={key_mitigation}')
        if key_mitigation:
            client.run_experiment(key_mitigation, backend=ibmqx2())
        if key_experiment:
            client.run_experiment(key_experiment, backend=ibmqx2())
