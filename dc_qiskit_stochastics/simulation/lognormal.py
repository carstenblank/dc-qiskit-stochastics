from multiprocessing import Pool
from typing import Tuple

import numpy as np


def sample_step(risk_free_interest, volatility, t_1, t_2, z):
    """
    Each single step is simply a log-normal with

    .. math::

        \mu = (r - 0.5 v^2) \cdot (t_2 - t_1)

        \sigma = v \cdot \sqrt{t_2 - t_1}

    where v is the volatility and r is the risk-free interest rate. This function then returns

    .. math::

        exp(mu + sigma * z)

    where z is a realization of a standard normal random variable.

    :param risk_free_interest: the risk-free interest rate on the market
    :param volatility: the volatility v > 0 of the asset
    :param t_1: the time point to start the step simulation
    :param t_2: the time point to stop the step simulation
    :param z: a realization of a standard normal random variable
    :return: returns the value of the asset.
    """
    mu = (risk_free_interest - 0.5 * volatility**2) * (t_2 - t_1)
    sigma = volatility * np.sqrt(t_2 - t_1)

    log_r = mu + sigma * z

    return np.exp(log_r)


def sample_path(risk_free_interest, volatility, start_value, time_steps: np.ndarray):
    """
    Calculate a sample path of length `n` given parameters.

    :param risk_free_interest: the risk-free interest rate
    :param volatility: volatility of the asset, always greater zero!
    :param start_value: start value of the asset
    :param time: the simulation time in total
    :param n: number of intermediate steps
    :return: a path of n realizations of increments.
    """
    n = time_steps.shape[0]
    Z = list(np.random.standard_normal(n))

    # Add the first evaluation point, which is the starting value
    time_steps = [0.0] + time_steps.tolist()

    path_elements = [start_value]
    for [t_1, t_2] in zip(time_steps, time_steps[1:]):
        z = Z.pop()
        s = sample_step(risk_free_interest, volatility, t_1, t_2, z)
        path_elements.append(s)

    # Skip the first value that we added above!
    path = []
    for i in range(1, len(path_elements)):
        s = np.prod(path_elements[0:i + 1])
        path.append(s)

    return path


def sample_last(risk_free_interest, volatility, start_value, time: Tuple[float, float], n: int):
    """
    Helper function to sample n steps of a log-normal distributed random variable.

    :param risk_free_interest: risk-free interest rate
    :param volatility: the volatility greater 0!
    :param start_value: start value of the asset
    :param time: duration of the simulation
    :param n: number of steps in between
    :return: returns the value at `time` of the assset given one simulation!
    """
    path = sample_path(risk_free_interest, volatility, start_value, time, n)
    return path[-1]


if __name__ == "__main__":

    risk_free_interest = 0.02
    volatility = 0.05
    start_value = 80
    time = (0, 10)
    n = 10

    arguments = [risk_free_interest, volatility, start_value, time, n]

    end_value_interest = start_value * np.exp(risk_free_interest * (time[1] - time[0]))

    samples = [arguments] * 10000

    with Pool() as pool:
        last = pool.starmap(sample_last, samples)
        print(f'{np.average(last)}/{end_value_interest}')


    # path = sample_path(risk_free_interest=0.02, volatility=0.05, start_value=100, time=(10, 20), n=10)
    # print(np.average(path))
    # print(path)
