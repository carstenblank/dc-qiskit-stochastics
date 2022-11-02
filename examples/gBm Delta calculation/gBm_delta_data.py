import numpy as np

S_0 = 100
r = 0.02
time_evaluation = 1
time_of_maturity = 10
time_to_maturity = time_of_maturity - time_evaluation
mu = 0.00
sigma = 0.02
K = np.round(S_0 * np.exp(r * time_to_maturity), decimals=4)