import numpy as np


def compute_cos_at(t):
    def inner(scale):
        return np.cos(scale*t)
    return inner


def compute_sin_at(t):
    def inner(scale):
        return np.sin(scale*t)
    return inner


def compute_with_char_func(coefficients: np.ndarray, period: float, cos, sin) -> np.ndarray:
    fourier_vector = get_fourier_vector_with_cos_sin(len(coefficients)//2, period, cos, sin)
    return fourier_vector.dot(coefficients)


def get_fourier_vector_with_cos_sin(n: int, period: float, cos, sin) -> np.ndarray:
    return np.asarray([
        cos(2 * np.pi * j / period) + 1.0j * sin(2 * np.pi * j / period)
        for j in range(-n, n + 1)
    ])


def get_fourier_vector_with_exp(n: int, period: float, exp) -> np.ndarray:
    return np.asarray([exp(2j * np.pi * j / period) for j in range(-n, n + 1)])


def get_coefficients(function: np.ndarray):
    dft = np.fft.fft(function) / len(function)
    coefficients = np.roll(dft, len(dft)//2)[1:]
    return coefficients
