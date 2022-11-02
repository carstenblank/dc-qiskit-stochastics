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
import logging
from datetime import datetime
from typing import List, Union, Optional, Callable

import numpy as np
import qiskit
import scipy
from qiskit.circuit import Parameter
from qiskit.providers.ibmq import IBMQBackend
from qiskit.transpiler import PassManager
from scipy import sparse

from dc_quantum_scheduling import PreparedExperiment
from . import get_default_pass_manager
from .dsp_common import apply_level, apply_initial, x_measurement, y_measurement
from .dsp_independent import index_independent_prep
from .dsp_util import create_qobj, extract_evaluations
from .qiskit_util import qasm_simulator

LOG = logging.getLogger(__name__)


class DiscreteStochasticProcess(object):
    realizations: np.array
    probabilities: np.array
    initial_value: float

    def __init__(self, initial_value: float, probabilities: np.ndarray, realizations: np.ndarray):
        self.initial_value = initial_value
        self.probabilities = probabilities
        self.realizations = realizations

    @staticmethod
    def _get_circuit_name(scaling: float) -> str:
        return f'dsp_simulation({scaling})'

    @staticmethod
    def _get_circuit_name_cos(scaling: float) -> str:
        return f'{DiscreteStochasticProcess._get_circuit_name(scaling)}_cosine'

    @staticmethod
    def _get_circuit_name_sin(scaling: float) -> str:
        return f'{DiscreteStochasticProcess._get_circuit_name(scaling)}_sine'

    def _proposition_one_circuit(self, scaling: Parameter, level_func=None, index_state_prep=None, with_barrier=False, **kwargs):
        # per default we use the standard Moettoennen level function
        level_func = apply_level if level_func is None else level_func
        # per default we use the independent index state preparation (also Moettoennen)
        index_state_prep = index_independent_prep if index_state_prep is None else index_state_prep

        LOG.debug(f"Data: initial value={self.initial_value}, "
                  f"probabilities={list(self.probabilities)}, realizations={list(self.realizations)},"
                  f"applied function={level_func.__name__}.")

        qc = qiskit.QuantumCircuit()

        LOG.debug(f"Initializing with {self.initial_value} and scaling {scaling}.")
        init_qc = apply_initial(self.initial_value, scaling)
        qc.extend(init_qc)
        if with_barrier:
            qc.barrier()

        for level, (p, r) in enumerate(zip(self.probabilities, self.realizations)):
            LOG.debug(f"Adding level {level}: {p} with {r} and scaling {scaling}.")
            qc_index = index_state_prep(level, p, **kwargs)
            qc_level_l = level_func(level, r, scaling, **kwargs)
            qc.extend(qc_index)
            if with_barrier:
                qc.barrier()
            qc.extend(qc_level_l)
            if with_barrier:
                qc.barrier()

        return qc

    def expval_cos_circuit(self, scaling: Parameter, level_func=None, index_state_prep=None, **kwargs):
        LOG.info(f'Cosine Circuit generation with scaling {scaling} and level_func={level_func}')
        qc = self._proposition_one_circuit(scaling, level_func, index_state_prep, **kwargs)
        qc.extend(x_measurement())

        qc.name = f'{qc.name}_cosine'

        LOG.debug(f"Circuit:\n{qc.draw(output='text', fold=-1)}")

        return qc

    def expval_sin_circuit(self, scaling: Parameter, level_func=None, index_state_prep=None, **kwargs):
        LOG.info(f'Sine Circuit generation with scaling {scaling} and level_func={level_func}')
        qc = self._proposition_one_circuit(scaling, level_func, index_state_prep, **kwargs)
        qc.extend(y_measurement())

        qc.name = f'{qc.name}_sine'

        LOG.debug(f"Circuit:\n{qc.draw(output='text', fold=-1)}")

        return qc

    def characteristic_function(self, evaluations: Union[List[float], np.ndarray, scipy.sparse.dok_matrix],
                                external_id: Optional[str] = None, level_func=None, pm: Optional[PassManager] = None,
                                transpiler_target_backend: Optional[Callable[[], IBMQBackend]] = None,
                                other_arguments: dict = None) -> PreparedExperiment:
        other_arguments = {} if other_arguments is None else other_arguments
        backend = transpiler_target_backend() if transpiler_target_backend is not None else qasm_simulator()
        if external_id is None:
            tags = other_arguments.get('tags', [])
            external_id = datetime.now().strftime('%Y%m%d-%H%M%S') + f"-{type(self).__name__}-{'-'.join(tags)}"

        scaling_v: Parameter = Parameter('v')
        qc_cos_param: qiskit.QuantumCircuit = self.expval_cos_circuit(scaling_v, level_func, **other_arguments)
        qc_sin_param: qiskit.QuantumCircuit = self.expval_sin_circuit(scaling_v, level_func, **other_arguments)

        pre_pm = None
        if pm is None:
            LOG.info('No PassManager given, trying to attain one.')
            pre_pm, pm = get_default_pass_manager(
                transpiler_target_backend=backend,
                other_arguments=other_arguments
            )

        qobj_list = create_qobj(qc_cos_param, qc_sin_param, scaling_v, evaluations=evaluations,
                                pass_manager=pm, pre_pass_manager=pre_pm, qobj_id=external_id,
                                other_arguments=other_arguments,
                                transpiler_target_backend=backend)

        prepared_experiment = PreparedExperiment(
            external_id=external_id,
            tags=other_arguments.get('tags', []),
            arguments=evaluations,
            parameters={
                'initial_value': self.initial_value,
                'probabilities': self.probabilities,
                'realizations': self.realizations,
                'pass_manager': pm.passes(),
                'qc_cos': qc_cos_param,
                'qc_sin': qc_sin_param,
                **other_arguments
            },
            qobj_list=qobj_list,
            callback=extract_evaluations,
            transpiler_backend=backend
        )

        return prepared_experiment
