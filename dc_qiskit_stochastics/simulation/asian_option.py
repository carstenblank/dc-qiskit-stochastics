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

import numpy as np
from scipy.stats import lognorm

logging.basicConfig(format=f'%(asctime)s::{logging.BASIC_FORMAT}', level='ERROR')
LOG = logging.getLogger(__name__)


class StateMachineDescription:

    realizations: np.ndarray
    probabilities: np.ndarray
    initial_value: float

    def __init__(self, initial_value: float, probabilities: np.ndarray, realizations: np.ndarray):
        self.initial_value = initial_value
        self.probabilities = probabilities
        self.realizations = realizations


class AsianOptionPricing:

    def __init__(self, s0, sigma, mu, time_steps, discretization):
        self.s0 = s0
        self.sigma = sigma
        self.mu = mu
        self.time_steps = time_steps
        self.discretization = discretization
        self.n = time_steps.shape[0]

    def compute_states(self, t: float) -> np.ndarray:
        s = self.sigma * np.sqrt(t)
        mu_tilde = (self.mu - 0.5 * self.sigma ** 2) * t + np.log(self.s0)
        scale = np.exp(mu_tilde)
        q = np.linspace(0.01, 0.99, self.discretization + 1)
        xq = lognorm.ppf(q, s=s, scale=scale)
        return xq

    def compute_probability(self, current_state: float, target_states: np.ndarray, delta_t: float) -> np.ndarray:
        s = self.sigma * np.sqrt(delta_t)
        mu_tilde = (self.mu - 0.5 * self.sigma ** 2) * delta_t + np.log(current_state)
        scale = np.exp(mu_tilde)
        _p = lognorm.pdf(target_states, s=s, scale=scale)
        return _p

    def get_state_machine_model(self) -> StateMachineDescription:
        delta_t = self.time_steps[-1] / self.n

        last_states = np.asarray([self.s0])
        states = {0: last_states}
        transition_matrices = {}
        for level in range(1, self.n + 1):
            # The target states are discretization + 1 as we are interested in the intervals
            intervals = self.compute_states(level * delta_t)
            target_states = intervals[:-1]

            transition_matrix = np.zeros(shape=(last_states.shape[0], target_states.shape[0]))
            for row, cs in enumerate(last_states):
                density_eval = self.compute_probability(cs, target_states, delta_t)
                zipped = np.vstack(
                    [target_states, intervals[1:], density_eval]
                )
                probabilities = zipped[2, :] * (zipped[1, :] - zipped[0, :])
                probabilities = probabilities / np.sum(probabilities)
                transition_matrix[row, :] = probabilities

            transition_matrices[level] = transition_matrix
            states[level] = target_states
            last_states = target_states

        return StateMachineDescription(
            initial_value=0,
            probabilities=np.asarray(list(transition_matrices.values())),
            realizations=np.asarray(list(states.values())[1:])/self.n
        )
