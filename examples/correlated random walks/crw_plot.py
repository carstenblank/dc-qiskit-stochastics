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
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

font = {'size': 20}
matplotlib.rc('font', **font)

if __name__ == "__main__":
    data: pd.DataFrame = pd.read_csv(
        f'./data/crw-4-fourier-qc-ibmqx2-P=100-L=100-v004-shots=32768.csv',
        usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9], index_col='evaluations'
    )

    fig: Figure = plt.figure(figsize=(8, 8))
    # ax: Axes = fig.add_axes([0.21, 0.16, 0.75, 0.7])
    ax: Axes = fig.add_subplot()

    ax.scatter(x=data['expectation value (sim/real)'],
               y=data['expectation value (sim/imag)'],
               color='blue', marker='.', s=30, label='simulation')
    ax.scatter(x=data['expectation value (real w/ mitigation)'],
               y=data['expectation value (imag w/ mitigation)'],
               color='red', marker='1', s=60, label='experiment')
    # ax.scatter(x=data['expectation value (real)'],
    #            y=data['expectation value (imag)'],
    #            color='red', marker='1', s=40, label='experiment w/o mitigation')
    ax.scatter(x=data['expectation value (theory/real)'],
               y=data['expectation value (theory/imag)'],
               color='black', marker='x', s=30, label='theory')
    ax.set_xlabel('$Re \\varphi (v)$')
    ax.set_ylabel('$Im \\varphi (v)$')
    ax.margins(0.05, 0.1)
    # ax.set_title(f'Evaluations of Characteristic Function')
    fig.legend(
        loc='upper right',  ncol=3, columnspacing=0.1,
        handletextpad=0.1,
        markerscale=2,
        labelspacing=0.1,
        framealpha=1.0,
        shadow=True
    )
    fig.show()
    fig.savefig(f'./images/crw-comparison-experiment-ibmqx2.png')
    fig.savefig(f'./images/crw-comparison-experiment-ibmqx2.svg')
    fig.savefig(f'./images/crw-comparison-experiment-ibmqx2.pdf')
