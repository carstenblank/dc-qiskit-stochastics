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

from dc_qiskit_stochastics.dsp_common import apply_level_two_realizations, apply_level

testing_data = [(
        1.0,
        0,
        np.asarray([
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
        ]),
        np.asarray([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]),
        apply_level_two_realizations
    ), (
        1.0,
        5.0,
        np.asarray([
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
        ]),
        np.asarray([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]),
        apply_level_two_realizations
    ), (
        1.0,
        0,
        np.asarray([
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
        ]),
        np.asarray([
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
        ]),
        apply_level_two_realizations
    ), (
        0.3333,
        4.0,
        np.asarray([
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
        ]),
        np.asarray([
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
        ]),
        apply_level_two_realizations
    ), (
        0.3333,
        4.0,
        np.asarray([
            [0.1, 0.9],
            [0.8, 0.2],
            [0.3, 0.7],
            [0.5, 0.5],
        ]),
        np.asarray([
            [-1.0, 1.0],
            [-0.4, 0.5],
            [-2.3, 1.1],
            [-1.0, 0.1],
        ]),
        apply_level_two_realizations
    ), (
        1.0,
        0,
        np.asarray([
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
        ]),
        np.asarray([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]),
        apply_level
    ), (
        1.0,
        5.0,
        np.asarray([
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
        ]),
        np.asarray([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]),
        apply_level
    ), (
        1.0,
        0,
        np.asarray([
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
        ]),
        np.asarray([
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
        ]),
        apply_level
    ), (
        0.3333,
        4.0,
        np.asarray([
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
        ]),
        np.asarray([
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
        ]),
        apply_level
    ), (
        0.3333,
        4.0,
        np.asarray([
            [0.1, 0.9],
            [0.8, 0.2],
            [0.3, 0.7],
            [0.5, 0.5],
        ]),
        np.asarray([
            [-1.0, 1.0],
            [-0.4, 0.5],
            [-2.3, 1.1],
            [-1.0, 0.1],
        ]),
        apply_level
    )
    ]

testing_data_state_machine = [
    {
        'initial_value': 10,
        'probabilities': np.asarray([
            [
                [0.5, 0.5]
            ],
            [
                [0.5, 0.5],
                [0.5, 0.5]
            ],
            [
                [0.5, 0.5],
                [0.5, 0.5]
            ],
            [
                [0.5, 0.5],
                [0.5, 0.5]
            ]
        ]),
        'realizations': np.asarray([
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1]
        ])
    },
    {
        'initial_value': 5,
        'probabilities': np.asarray([
            [
                [0.1, 0.6, 0.3, 0.0]
            ],
            [
                [0.1, 0.6, 0.3, 0.0],
                [0.5, 0.3, 0.1, 0.1],
                [0.1, 0.1, 0.2, 0.6],
                [0.3, 0.0, 0.4, 0.3]
            ],
            [
                [0.1, 0.6, 0.3, 0.0],
                [0.5, 0.3, 0.1, 0.1],
                [0.1, 0.1, 0.2, 0.6],
                [0.3, 0.0, 0.4, 0.3]
            ]
        ]),
        'realizations': np.asarray([
            [-2, -1, 1, 2],
            [-2, -1, 1, 2],
            [-2, -1, 1, 2]
        ])
    }
]