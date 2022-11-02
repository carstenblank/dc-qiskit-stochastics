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

S_0 = 100
r = 0.02
time_evaluation = 1
time_of_maturity = 10
time_to_maturity = time_of_maturity - time_evaluation
mu = 0.00
sigma = 0.02
K = np.round(S_0 * np.exp(r * time_to_maturity), decimals=4)