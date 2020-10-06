import logging
import unittest
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dc_qc_random_walks.benchmark import monte_carlo_fbm
from dc_qc_random_walks.plotting import plot_characteristic_function
from dc_qc_random_walks.qiskit.fractional_bownian_motion import FractionalBrownianMotion
from dc_qc_random_walks.simulation_scheduling import PreparedExperiment, processor
from fbm import FBM

logging.basicConfig(format=logging.BASIC_FORMAT, level='INFO')
LOG = logging.getLogger(__name__)


def do_experiment_with_pickle(experiment: PreparedExperiment):
    return processor.execute_simulation(experiment).wait().serialize()


class FractionalBrownianMotionTests(unittest.TestCase):

    def test_benchmark(self):
        density = lambda H, p: (1 - H) * np.power(2, 3 - 2 * H) * np.power(1 - p, 1 - 2 * H)
        # density = lambda H, p: 2
        hurst_index = 0.5
        initial_value = 0.0
        approximation_N = 40
        approximation_M = 10000  # int(np.ceil(np.power(approximation_N, 2.8 - 2 * hurst_index)))
        time_evaluation = 10.00
        mc_samples = 200
        mc_samples_benchmark = 10000

        L = 40
        P = 100

        LOG.info(f'Hurst={hurst_index}, N={approximation_N}, M={approximation_M}...')

        fbm = FractionalBrownianMotion(
            hurst_index=hurst_index, initial_value=initial_value, approximation_N=approximation_N,
            approximation_M=approximation_M, density=density,
            time_evaluation=time_evaluation
        )
        # fbm.sample(density_descretization=10000, debug_plot=True)
        fbm.prepare_integral(density_descretization=approximation_M, debug_plot=True)
        l_array = np.arange(-L, L + 1)
        evaluations = 2 * np.pi / P * l_array

        # LOG.info('Benchmarking now...')
        # start = datetime.now()
        # # ==== This is the characteristic function of the standard Brownian motion ====
        perfect = np.exp(- np.power(evaluations, 2) * time_evaluation / 2)
        # # ==== The monte carlo simulation of the fBm ====
        # fbm_benchmark = monte_carlo_fbm(
        #     hurst=hurst_index,
        #     time_evaluation=time_evaluation,
        #     steps=approximation_N,
        #     samples=mc_samples_benchmark,
        #     func=lambda x: np.exp(1.0j * x),
        #     evaluations=evaluations,
        #     offset=0.0,
        #     use_multiprocessing=True,
        #     method='hosking'
        # )
        # end = datetime.now()
        # LOG.info(f"Benchmarking done in {end - start}!")

        LOG.info(f"Starting 'experiment' now.")
        char_func_eval = fbm.benchmark(
            evaluations=evaluations,
            other_arguments={
                'monte_carlo_samples': mc_samples,
                'force_single_threaded': False,
                'mulitprocessing_processes': 8
            }
        )
        LOG.info("'Experiment' Done!")

        # prep_list = fbm.characteristic_function_experiments(evaluations=evaluations, other_arguments={'shots': 8192})

        # LOG.info(f'Starting in parallel {len(prep_list)} prepared experiments.')
        # with Pool() as pool:
        #     fin_list_pickled = pool.map(do_experiment_with_pickle, prep_list)
        # LOG.info(f'Finished all {len(prep_list)} prepared experiments.')

        # fin_list = [FinishedExperiment.deserialize(b) for b in fin_list_pickled]
        # char_func_eval = fbm.characteristic_function(fin_list, fitter_experiments=[])

        # LOG.info(char_func_eval)

        # varphi = characteristic_function_fbm(
        #     hurst=hurst_index,
        #     time_evaluation=time_evaluation,
        #     steps=10,
        #     samples=100000,
        #     use_multiprocessing=False
        # )
        # monte_carlo = varphi(evaluations)

        title = f'H={hurst_index}, N={approximation_N}, M={approximation_M}, s={mc_samples}, t={time_evaluation}'
        # FIXME: not the best plot
        # fig = plot_characteristic_function(
        #     simulation=fbm_benchmark,
        #     # simulation=[],
        #     experiment=char_func_eval,
        #     # experiment=[],
        #     theory=perfect,
        #     title=f'\nH={hurst_index}, N={approximation_N}, M={approximation_M}, s={mc_samples}, t={time_evaluation}'
        # )
        # fig.axes[0].set_xlim(-0.1, 1.1)
        # fig.axes[0].set_ylim(-0.6, 0.6)
        # fig.show()

        df = pd.DataFrame(index=evaluations)
        # df['simulation'] = fbm_benchmark
        df['experiment'] = char_func_eval
        df['theory'] = perfect

        # df['simulation'].map(lambda x: x.real).plot(linestyle='-', c='blue')
        # df['simulation'].map(lambda x: x.imag).plot(linestyle=':', c='blue')
        df['experiment'].map(lambda x: x.real).plot(linestyle='-', c='red')
        df['experiment'].map(lambda x: x.imag).plot(linestyle=':', c='red')
        df['theory'].map(lambda x: x.real).plot(linestyle='-', c='black')
        df['theory'].map(lambda x: x.imag).plot(linestyle=':', c='black')
        plt.title(title)
        plt.show()
        plt.savefig(f'../../images/fBm_benchmark_{title.replace(", ", "-")}.pdf')
        df.to_csv(f'../../data/fBm_benchmark_{title.replace(", ", "-")}.csv')

        LOG.info("\n" + df.to_string(float_format=lambda x: f'{x:.6f}'))
