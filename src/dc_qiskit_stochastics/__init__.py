import logging
from typing import List

import numpy as np
from qiskit.providers import BaseBackend
from qiskit.providers.models import BackendConfiguration
from qiskit.transpiler import PassManagerConfig, CouplingMap
from qiskit.transpiler.preset_passmanagers import level_3_pass_manager, level_0_pass_manager

# TODO: extract to separate files once POC stage is done

LOG = logging.getLogger(__name__)


def get_default_pass_manager(transpiler_target_backend: BaseBackend, other_arguments: dict, with_pre_pm=True):

    # noinspection PyTypeChecker
    config = PassManagerConfig(
        initial_layout=None,
        basis_gates=None,
        coupling_map=None,
        layout_method='trivial',  # FIXME: once qiskit is fixed, we can be more free
        routing_method='lookahead',
        backend_properties=None,
        seed_transpiler=None
    )

    backend_config: BackendConfiguration = transpiler_target_backend.configuration()
    config.basis_gates = backend_config.basis_gates
    config.coupling_map = CouplingMap(couplinglist=backend_config.coupling_map) if backend_config.coupling_map is not None else None
    config.backend_properties = backend_config

    if 'initial_layout' in other_arguments: config.initial_layout = other_arguments['initial_layout']
    if 'basis_gates' in other_arguments: config.basis_gates = other_arguments['basis_gates']
    if 'coupling_map' in other_arguments: config.coupling_map = other_arguments['coupling_map']
    if 'layout_method' in other_arguments: config.layout_method = other_arguments['layout_method']
    if 'routing_method' in other_arguments: config.routing_method = other_arguments['routing_method']
    if 'backend_properties' in other_arguments: config.backend_properties = other_arguments['backend_properties']
    if 'seed_transpiler' in other_arguments: config.seed_transpiler = other_arguments['seed_transpiler']

    pre_pass_manager = level_0_pass_manager(config)
    pass_manager = level_3_pass_manager(config)
    LOG.info(f'Created PassManager {pass_manager} with config {config}.')
    if with_pre_pm:
        return pre_pass_manager, pass_manager
    else:
        return pass_manager


def qobj_mapping(shots: int, max_shots: int, max_experiments_per_qobj: int, evaluations_count: int) -> np.array:
    """
    example: len(evaluations) = 12, max_shots = 8192, shots = 40960, max_experiments (per qobj) = 3
    experiments to reach shots = 5
    input   : qobj index
    [0] :  0  0  0  1  1
    [1] :  1  2  2  2  3
    [2] :  3  3  4  4  4
    [3] :  5  5  5  6  6
    [4] :  6  7  7  7  8
    [5] :  8  8  9  9  9
    [6] : 10 10 10 11 11
    [7] : 11 12 12 12 13
    [8] : 13 13 14 14 14
    [9] : 15 15 15 16 16
    [10] : 16 17 17 17 18
    [11] : 18 18 19 19 19

    :param max_shots: max shots per experiment
    :param max_experiments_per_qobj: max experiments per qobj
    :param evaluations_count: the evaluations requested
    :return: a matrix with qobj index
    """
    rows = evaluations_count
    columns = int(np.ceil(shots / max_shots))

    qobj_index = 0
    experiment_index = 0
    matrix: List[List[int]] = []
    for r in range(rows):
        matrix_row: List[int] = []
        for c in range(columns):
            matrix_row.append(qobj_index)
            experiment_index = experiment_index + 1

            # update if maximum experiments per qobj is reached!
            if experiment_index >= max_experiments_per_qobj:
                experiment_index = 0
                qobj_index = qobj_index + 1

        matrix.append(matrix_row)

    return np.asarray(matrix)
