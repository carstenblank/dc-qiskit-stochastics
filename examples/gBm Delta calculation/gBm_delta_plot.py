import os.path

import pandas as pd
from matplotlib.axes import Axes

if __name__ == "__main__":
    experiment:pd.DataFrame = pd.read_csv(
        # '../../data/gBm-delta-donsker-4-fourier-qc-ibmq_ourense-P=100-L=100.csv',
        './data/gBm-delta-donsker-4-fourier-qc-ibmqx2-P=100-L=100.csv',
        index_col='strike price', usecols=[1, 2, 3, 4, 5]
    )
    theory_benchmark:pd.DataFrame = pd.read_csv(
        './data/gBm-delta-theory-fourier-P=100-L=100-n=4.csv',
        index_col='strike price',
        usecols=[1, 2, 3]
    )
    theory:pd.DataFrame = pd.read_csv(
        './data/gBm-delta-theory-fourier-P=10000-L=10000-n=10.csv',
        index_col='strike price',
        usecols=[1, 2, 3]
    )

    # df = theory \
    #     .join(theory_benchmark, lsuffix='theory', rsuffix=' benchmark') \
    #     .join(experiment, rsuffix=' experiment')
    # df.plot()
    ax: Axes = theory.reset_index().plot(x='strike price', y='expectation value (real)',
                                         style=':', color='black', label='Delta')
    theory_benchmark.reset_index().plot(kind='line', ax=ax,
                                        x='strike price', y='expectation value (real)',
                                        color='black', label='approx. Delta (real)')
    theory_benchmark.reset_index().plot(kind='line', ax=ax,
                                        x='strike price', y='expectation value (imag)',
                                        color='blue', label='approx. Delta (imag)')
    experiment.reset_index().plot(kind='scatter', ax=ax,
                                  x='strike price', y='expectation value (real)',
                                  marker='x', s=60, color='black', label='approx. QC Delta (real)')
    experiment.reset_index().plot(kind='scatter', ax=ax,
                                  x='strike price', y='expectation value (imag)',
                                  marker='1', s=60, color='blue', label='approx. QC Delta (imag)')
    experiment.reset_index().plot(kind='scatter', ax=ax,
                                  x='strike price', y='expectation value (sim/real)',
                                  marker='.', s=40, color='black', label='approx. sim-QC Delta (real)')
    experiment.reset_index().plot(kind='scatter', ax=ax,
                                  x='strike price', y='expectation value (sim/imag)',
                                  marker='^', s=40, color='blue', label='approx. sim-QC Delta (imag)')
    ax.set_ylabel('expectation value of Delta (real/imag)')
    ax.set_title('')
    ax.figure.show()
    if os.path.exists("./images"):
        ax.figure.savefig(f'./images/delta-comparison-experiment-ibmqx2.png')
        ax.figure.savefig(f'./images/delta-comparison-experiment-ibmqx2.svg')
        ax.figure.savefig(f'./images/delta-comparison-experiment-ibmqx2.pdf')
