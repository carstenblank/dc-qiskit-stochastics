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
