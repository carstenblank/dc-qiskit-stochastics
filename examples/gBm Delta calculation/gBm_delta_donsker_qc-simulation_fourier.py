import datetime
import logging

import dc_quantum_scheduling as scheduling
import matplotlib.pyplot as plt
import numpy as np
from dc_quantum_scheduling import processor, client
from dc_quantum_scheduling.qiskit.qiskit_provider import set_provider_config, provider

from dc_qiskit_stochastics.discrete_stochastic_process import DiscreteStochasticProcess
from dc_qiskit_stochastics.error_mitigation import create_error_mitigation_experiment
from gBm_delta_data import S_0, r, time_evaluation, time_of_maturity, time_to_maturity, mu, sigma, K

logging.basicConfig(format=logging.BASIC_FORMAT, level='INFO')
[logging.getLogger(name).setLevel('WARN') for name in logging.root.manager.loggerDict if name.startswith('qiskit')]
logging.getLogger(__name__).setLevel('INFO')

LOG = logging.getLogger(__name__)


set_provider_config(hub='ibm-q', group='open', project='main')
client.set_url('http://localhost:8080')


if __name__ == "__main__":

    version = '001'

    n = 4

    shots = 8192 * 10

    P = 100
    L = 100

    backend = lambda: provider().get_backend("ibmq_lima")
    backend_name = "sim" if backend is None else backend().name()

    l_array = np.arange(-L, L + 1)
    fourier_coefficients = np.asarray(
        [-1.0j / (2 * np.pi * l) * np.exp(-2 * np.pi ** 2 * l ** 2 / P ** 2) if l != 0 else 1 / 2
         for l in l_array])
    char_func_input = 2 * np.pi / P * l_array

    strike_prices = np.arange(10, 200, 5)
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
            'tags': ['gBm_delta_donsker_qc_fourier', f'{datetime.datetime.now().date().isoformat()}--{version}',
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
        prepared_mitigation = create_error_mitigation_experiment(exp)
        prepared_experiments.append((exp, prepared_mitigation))

        LOG.info(f'=============== Finished Preparation for STRIKE_PRICE={strike_price} =====================')

    exp_vals = []
    for exp, prepared_mitigation in prepared_experiments:

        strike_price = exp.parameters["strike_price"]

        # Error mitigation
        running_mitigation = processor.execute_simulation(prepared_mitigation)
        LOG.info(f'Mitigation running for {strike_price}')
        finished_mitigation = running_mitigation.wait()
        LOG.info(f'Mitigation done for {strike_price}')

        # Run the experiment
        running_experiment = processor.execute_simulation(exp)
        LOG.info(f'Experiment running for {strike_price}')
        finished_exp = running_experiment.wait()
        LOG.info(f'Experiment done for {strike_price}')

        output = finished_exp.get_output(finished_mitigation.get_output())
        char_func_output = np.asarray(output)
        exp_val = char_func_output.dot(fourier_coefficients)

        LOG.info(f'For strike price {strike_price} the evaluation of the expectation value is {exp_val}.')

        exp_vals.append(exp_val)

    expectation_approx = np.asarray(exp_vals)

    plt.scatter(x=strike_prices, y=np.real(expectation_approx))
    plt.scatter(x=strike_prices, y=np.imag(expectation_approx))
    plt.title(
        f'gBm: $S_0$={S_0}/K={K}/r={r}/$\mu$={mu}/$\sigma$={sigma}/$t$={time_evaluation},/$T$={time_of_maturity}\n'
        f'RW-QC (Donkser), n={n}, P={P}, L={L}\n'
        f'backend: shots={shots}, transpiler={backend_name}, exec=sim')
    plt.savefig(f'../images/gBm_delta_donsker_fourier_sim_withnoise_ourense.pdf')
    plt.savefig(f'../images/gBm_delta_donsker_fourier_sim_withnoise_ourense.eps')
    plt.savefig(f'../images/gBm_delta_donsker_fourier_sim_withnoise_ourense.png')
    plt.show()


    # initial_value = finished_exp.parameters.get('initial_value')
    # probabilities = finished_exp.parameters.get('probabilities')
    # realizations = finished_exp.parameters.get('realizations')
    # evaluations = finished_exp.arguments
    # varphi = benchmark.characteristic_function_rw_ind(initial_value, probabilities, realizations)
    # benchmark.monte_carlo_fbm(hurst=0.5, time_evaluation=1)

    # fig = plot_characteristic_function(
    #     simulation=output,
    #     experiment=[],
    #     theory=varphi(evaluations),
    #     title=f'gBm via Donsker\n'
    #           f'$S_0$={S_0}/K={K}/r={r}/$\mu$={mu}/$\sigma$={sigma}/$T-t$={time_to_maturity}, n={n}/P={P}/L={L}\n'
    #           f'{backend_name}, shots={shots}, err.mit., w/ noise={not configuration.get("no_noise", False)}'
    # )
    # fig.show()

    # fig.savefig(f'../../images/{finished_exp.get_id()}.pdf')
    # fig.savefig(f'../../images/{finished_exp.get_id()}.eps')
    # fig.savefig(f'../../images/{finished_exp.get_id()}.png')

    # save('../../data/', simulation=finished_exp)
