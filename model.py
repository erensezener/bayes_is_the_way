import numpy as np
import matplotlib.pyplot as plt

# params = lambda: None
# params.p_common = 0.8
# params.sigma_v = 0.6
# params.sigma_a = 3.1
# params.sigma_p = 15
# params.mu_p = 0


def prior_C_equals_1(x_v, x_a, params):
    sigma_squared_sum = (params.sigma_v ** 2 * params.sigma_a ** 2
                         + params.sigma_v ** 2 * params.sigma_p ** 2
                         + params.sigma_a ** 2 * params.sigma_p ** 2)

    prefactor = 1. / (2 * np.pi * np.sqrt(sigma_squared_sum))

    exponent = -1. / 2 * (np.power(x_v - x_a, 2) * params.sigma_p ** 2
                          + np.power(x_v - params.mu_p, 2) * params.sigma_a ** 2
                          + np.power(x_a - params.mu_p, 2) * params.sigma_v ** 2) \
               / sigma_squared_sum
    return prefactor * np.exp(exponent)


def prior_C_equals_2(x_v, x_a, params):
    prefactor = 1. / (
        2 * np.pi * np.sqrt((params.sigma_v ** 2 + params.sigma_p ** 2) * (params.sigma_a ** 2 + params.sigma_p ** 2)))
    exponent = -1. / 2 * ((np.power(x_v - params.mu_p, 2) / (params.sigma_v ** 2 + params.sigma_p ** 2)
                           + np.power(x_a - params.mu_p, 2) / (params.sigma_a ** 2 + params.sigma_p ** 2)))
    return prefactor * np.exp(exponent)


def posterior_C_equals_1(x_v, x_a, params):
    return prior_C_equals_1(x_v, x_a, params) * params.p_common / \
           (prior_C_equals_1(x_v, x_a, params) * params.p_common
            + prior_C_equals_2(x_v, x_a, params) * (1 - params.p_common))


def estimated_s_v_when_C_equals_2(x_v, params):
    return (x_v / params.sigma_v * 2 + params.mu_p / params.sigma_p ** 2) / \
           (1. / params.sigma_v ** 2 + 1. / params.sigma_p ** 2)


def estimated_s_a_when_C_equals_2(x_a, params):
    return (x_a / params.sigma_a * 2 + params.mu_p / params.sigma_p ** 2) / \
           (1. / params.sigma_a ** 2 + 1. / params.sigma_p ** 2)


def estimated_s_when_C_equals_1(x_a, x_v, params):
    return (x_a / params.sigma_a * 2 + x_v / params.sigma_v * 2 + params.mu_p / params.sigma_p ** 2) / \
           (1. / params.sigma_a ** 2 + 1. / params.sigma_v ** 2 + 1. / params.sigma_p ** 2)


def estimated_s_v(x_v, x_a, params):
    return posterior_C_equals_1(x_v, x_a, params) * estimated_s_when_C_equals_1(x_a, x_v, params) + \
           (1 - posterior_C_equals_1(x_v, x_a, params)) * estimated_s_v_when_C_equals_2(x_v, params)


def estimated_s_a(x_v, x_a, params):
    return posterior_C_equals_1(x_v, x_a, params) * estimated_s_when_C_equals_1(x_a, x_v, params) + \
           (1 - posterior_C_equals_1(x_v, x_a, params)) * estimated_s_a_when_C_equals_2(x_a, params)
