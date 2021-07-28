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