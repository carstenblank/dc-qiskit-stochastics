import os

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

font = {'size': 26}
matplotlib.rc('font', **font)

if __name__ == "__main__":
    data: pd.DataFrame = pd.read_csv(
        # '../../data/gBm-delta-donsker-4-fourier-qc-details-ibmq_ourense-P=100-L=100-K=110.0.csv',
        './data/gBm-delta-donsker-4-fourier-qc-details-ibmqx2-P=100-L=100-K=110.0.csv',
        index_col='evaluations', usecols=[1, 2, 3, 4, 5, 6, 7]
    )

    fig: Figure = plt.figure(figsize=(10, 8))
    fig.tight_layout()
    ax: Axes = fig.add_subplot()

    ax.scatter(x=data['sim (real)'],
               y=data['sim (imag)'],
               color='blue', marker='.', s=160, label='simulation')
    ax.scatter(x=data['exp (real)'],
               y=data['exp (imag)'],
               color='red', marker='1', s=240, label='experiment')
    ax.scatter(x=data['theory (real)'],
               y=data['theory (imag)'],
               color='black', marker='x', s=160, label='theory')
    ax.set_xlabel('$Re \\varphi (v)$')
    ax.set_ylabel('$Im \\varphi (v)$')
    ax.margins(0.05, 0.1)
    # ax.set_title(f'Evaluations of Characteristic Function for {title}')
    fig.legend(
        loc='upper right',  ncol=3, columnspacing=0.1,
        handletextpad=0.1,
        markerscale=1.5,
        labelspacing=0.1,
        framealpha=1.0,
        shadow=True
    )
    fig.show()
    if os.path.exists("./images"):
        fig.savefig(f'./images/delta-comparison-experiment-details-ibmqx2.png')
        fig.savefig(f'./images/delta-comparison-experiment-details-ibmqx2.svg')
        fig.savefig(f'./images/delta-comparison-experiment-details-ibmqx2.pdf')
