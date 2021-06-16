import itertools
import logging
from multiprocessing import Pool
from typing import List, Union, Optional, Tuple

import numpy as np
from fbm import FBM
from nptyping import NDArray

LOG = logging.getLogger(__name__)


def characteristic_function_rw_ind_two_step(initial_value: float, probabilities: np.ndarray, realizations: np.ndarray):
    def eval(p: float, r1: float, r2: float):
        return p * np.exp(1.0j * r1) + (1 - p) * np.exp((1.0j * r2))

    def _inner(evaluations: Union[float, List[float], np.ndarray]):
        if isinstance(evaluations, float):
            evaluations = np.asarray([evaluations])
        if isinstance(evaluations, list):
            evaluations = np.asarray(evaluations)
        outcomes = []
        for v in evaluations:
            single_char_func_eval_list = [eval(1.0, v * initial_value, 0.0)]
            for p, r in zip(list(probabilities), list(v * realizations)):
                value = eval(p[0], r[0], r[1])
                single_char_func_eval_list.append(value)

            outcomes.append(np.prod(single_char_func_eval_list))
        return outcomes

    return _inner


def characteristic_function_rw_ind(initial_value: float, probabilities: np.array, realizations: np.array,
                                   evaluations: Union[float, List[float], np.ndarray]) -> List[complex]:
    assert probabilities.shape == realizations.shape

    def eval(p_list: List[float], r_list: List[float]):
        return np.sum([p * np.exp(1.0j * r) for p, r in zip(p_list, r_list)])

    if isinstance(evaluations, float):
        evaluations = np.asarray([evaluations])
    if isinstance(evaluations, list):
        evaluations = np.asarray(evaluations)
    outcomes = []
    for v in evaluations:
        initial_probabilities = [1.0] + (probabilities.shape[0] - 1) * [0.0]
        initial_realizations = [v * initial_value] + (realizations.shape[0] - 1) * [0.0]
        single_char_func_eval_list = [eval(initial_probabilities, initial_realizations)]

        for p, r in zip(list(probabilities), list(v * realizations)):
            value = eval(p, r)
            single_char_func_eval_list.append(value)

        outcomes.append(complex(np.prod(single_char_func_eval_list)))
    return outcomes


def brute_force_rw_ind(probabilities: [NDArray or np.ndarray], realizations: [NDArray or np.ndarray],
                       initial_value: float, scaling: float, func=None, **kwargs) -> Optional[float]:
    func = func if func is not None else lambda x: np.exp(1.0j * x)
    levels, k = probabilities.shape
    if levels <= 20:
        results: List[float] = []
        for steps in itertools.product(range(k), repeat=levels):
            all_probs = probabilities[np.arange(len(probabilities)), steps]
            LOG.debug(f'Found the extraction probabilities to be {all_probs}.')
            all_reals = realizations[np.arange(len(realizations)), steps]
            LOG.debug(f'Found the extraction realizations to be {all_reals}.')

            p = np.prod(all_probs)
            x = initial_value + np.sum(all_reals)

            value = p * func(scaling * x)
            results.append(value)

        return sum(results)
    else:
        print("Brute Force skipped.")
        return None


def monte_carlo_rw_ind(mc_samples, choices, probs, length, scalings: np.ndarray, offset: float,
                       func=None) -> Optional[np.ndarray]:
    apply_func = func
    if isinstance(func, bytes):
        import dill
        apply_func = dill.loads(func)
    if mc_samples > 0:
        exp_vals = []
        for s in scalings:
            results = []
            for i in range(mc_samples):
                steps = np.random.choice(choices, p=probs, size=length)
                summed_up = offset + np.sum(steps)
                if apply_func is None:
                    position = np.exp(1.0j * s * summed_up)
                else:
                    from inspect import signature
                    sig = signature(apply_func)
                    if len(sig.parameters) == 1:
                        position = apply_func(s * summed_up)
                    else:
                        position = apply_func(s, summed_up)
                results.append(position)
            exp_val = np.mean(results)
            exp_vals.append(exp_val)
        return np.asarray(exp_vals)
    return None


def brute_force_correlated_walk(probabilities: [NDArray or np.ndarray], realizations: [NDArray or np.ndarray],
                                initial_value: float, scaling: float = 1.0, func=None, return_hist=False,
                                **kwargs) -> Optional[float or Tuple[float, np.ndarray, np.ndarray]]:

    assert probabilities.shape == realizations.shape

    func = func if func is not None else lambda x: np.exp(1.0j * x)

    if isinstance(func, bytes):
        import dill
        func = dill.loads(func)

    levels, k = realizations.shape
    assert k == 2
    path_probability = []
    path_values = []
    if levels <= 20:
        results: List[float] = []
        for steps in itertools.product(range(k), repeat=levels):
            # initial probability:
            assert np.sum(probabilities[0]) == 1
            all_probs = [probabilities[0][0] if steps[0] == 0 else probabilities[0][1]]

            # all other probabilities are taken from the last step!
            last_current_step: List[Tuple[int, int]] = list(itertools.zip_longest(steps[:-1], steps[1:]))
            all_probs = all_probs + [probabilities[i + 1][last_step]
                                     if last_step == current_step
                                     else 1 - probabilities[i + 1][last_step]
                                     for i, (last_step, current_step) in enumerate(last_current_step)]
            LOG.debug(f'Found the extraction probabilities to be {all_probs}.')

            all_reals = realizations[np.arange(len(realizations)), steps]
            LOG.debug(f'Found the extraction realizations to be {all_reals}.')

            p = np.prod(all_probs)
            x = initial_value + np.sum(all_reals)

            LOG.debug(f'P[X = {steps}, \Sigma = {x}] = {p:.16f}')

            path_probability.append(p)
            path_values.append(x)

            value = p * func(scaling * x)
            results.append(value)
        matrix = np.asarray(results)
        if return_hist:
            numbers = np.asarray(list(zip(path_values, path_probability)))
            unique_outcomes = np.unique(numbers[:, 0])
            histogram = np.asarray([(n, numbers[numbers[:, 0] == n, 1].sum()) for n in unique_outcomes])

            return matrix.sum(axis=0), histogram
        else:
            return matrix.sum(axis=0)
    else:
        print("Brute Force skipped.")
        return None


def _monte_carlo_correlated_walk_path(probabilities: [NDArray or np.ndarray], realizations: [NDArray or np.ndarray]):
    path = []
    steps = []
    for i, (p, r) in enumerate(zip(probabilities, realizations)):
        success = [np.random.binomial(1, p[0]) == 1, np.random.binomial(1, p[1]) == 1]
        if i == 0:
            path_increment = 0 if success[0] else 1
        else:
            last_path_increment = path[-1]
            path_increment = last_path_increment if success[last_path_increment] else 1 - last_path_increment
        step = r[path_increment]
        path.append(path_increment)
        steps.append(step)
    summed = np.sum(steps)
    return summed


def monte_carlo_correlated_walk(probabilities: np.ndarray, realizations: np.ndarray,
                                initial_value: float, scaling: [float or np.ndarray], samples: int = 100,
                                func=None) -> Optional[float or np.ndarray]:
    assert probabilities.shape == realizations.shape
    is_float_input = False
    if np.isscalar(scaling):
        is_float_input = True
        scaling = np.asarray([scaling])
    sample_list = [_monte_carlo_correlated_walk_path(probabilities, realizations) for _ in range(samples)]
    sample_vec = np.asarray(sample_list)
    sample_vec = initial_value + sample_vec
    scaled_vectors = np.tile(sample_vec, (scaling.size, 1)).transpose() * scaling
    sample_vec_func = np.vectorize(func or (lambda v: np.exp(1.0j * v)))(scaled_vectors)
    exp_val_approx: np.ndarray = np.average(sample_vec_func, axis=0)

    if is_float_input:
        return exp_val_approx[0]
    else:
        return exp_val_approx


def _cweval(f: FBM, scaling: float, offset: float = 0.0, func=None):
    fbm_sample = f.fbm()
    end_evaluation = offset + fbm_sample[-1]
    apply_func = func
    if isinstance(func, bytes):
        import dill
        apply_func = dill.loads(func)
    if apply_func is None:
        char_eval = np.exp(1.0j * scaling * end_evaluation)
    else:
        from inspect import signature
        sig = signature(apply_func)
        if len(sig.parameters) == 1:
            char_eval = apply_func(scaling * end_evaluation)
        else:
            char_eval = apply_func(scaling, end_evaluation)
    return char_eval


def _per_eval(f: FBM, e: float, samples: int, func=None, offset: float = 0.0):
    expected_value = [_cweval(f, e, offset, func) for _ in range(samples)]
    monte_carlo_eval = np.average(expected_value)
    return monte_carlo_eval


def monte_carlo_fbm(hurst: float, evaluations: np.ndarray, time_evaluation: float = 1.0, steps: int = 1024,
                    offset: float = 0.0, samples: int = 1024, use_multiprocessing: bool = False,
                    processes_multiprocessing=None, func=None, method='daviesharte'):
    f = FBM(n=steps, hurst=hurst, length=time_evaluation, method=method)

    apply_func = func
    if isinstance(func, bytes):
        import dill
        apply_func = dill.loads(func)

    if use_multiprocessing:
        import dill
        pickled_func = dill.dumps(apply_func)
        with Pool(processes=processes_multiprocessing) as pool:
            return pool.starmap(_per_eval, [(f, e, samples, pickled_func, offset) for e in evaluations])
    else:
        monte_carlo = []
        for e in evaluations:
            expected_value = [_cweval(f, e, offset, apply_func) for _ in range(samples)]
            monte_carlo_eval = np.average(expected_value)
            monte_carlo.append(monte_carlo_eval)
        return monte_carlo


def _asian_option_price(risk_free_interest, volatility, start_value, time_steps: np.ndarray):
    from dc_qiskit_stochastics.simulation.lognormal import sample_path
    path = sample_path(risk_free_interest, volatility, start_value, time_steps)
    value = np.average(path)
    return value


def char_func_asian_option(risk_free_interest, volatility, start_value, time_steps: np.ndarray, samples: int, evaluations: np.ndarray):
    arguments = [risk_free_interest, volatility, start_value, time_steps]
    samples_arguments = [arguments] * samples

    char_func_est_list = []
    with Pool() as pool:
        for v in evaluations:
            values = pool.starmap(_asian_option_price, samples_arguments)
            char_func_vec = np.exp(1.0j * v * np.asarray(values))
            # char_func_vec = np.cos(v * np.asarray(values))
            char_func_est = np.average(char_func_vec)
            char_func_est_list.append(char_func_est)

    return np.asarray(char_func_est_list)
