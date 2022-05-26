import numpy as np
from typing import List, Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_characteristic_function(simulation:  [List[complex] or np.ndarray],
                                 experiment: [List[complex] or np.ndarray],
                                 theory: [List[complex] or np.ndarray],
                                 title: Optional[str] = None) -> Figure:

    fig: Figure = plt.figure()
    ax: Axes = fig.add_subplot()

    if len(simulation) > 0:
        ax.scatter(x=[o.real for o in simulation],
                    y=[o.imag for o in simulation],
                    color='b', marker='x', label='simulation')
    if len(experiment) > 0:
        ax.scatter(x=[o.real for o in experiment],
                    y=[o.imag for o in experiment],
                    color='black', marker='x', label='experiment')
    if len(theory) > 0:
        ax.scatter(x=[o.real for o in theory],
                    y=[o.imag for o in theory],
                    color='r', marker='.', label='theory')
    ax.set_xlabel('$Re \\varphi (v)$')
    ax.set_ylabel('$Im \\varphi (v)$')
    if title is not None:
        ax.set_title(f'Evaluations of Characteristic Function for {title}')
    else:
        ax.set_title(f'Evaluations of Characteristic Function')
    fig.legend(loc='lower right')

    return fig
