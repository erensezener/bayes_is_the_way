import functools

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import emcee
import time

import model


N = int(1e4)

true_params = lambda: None
true_params.p_common = 0.8
true_params.sigma_v = 0.6
true_params.sigma_a = 3.1
true_params.sigma_p = 15
true_params.mu_p = 0

cs = np.sign(np.random.rand(N) - true_params.p_common) / 2. + 1.5  # only consists of 1s and 2s
is_class_1 = np.isclose(cs, 1.)
s_samples = np.random.normal(true_params.mu_p, true_params.sigma_p, (N, 2))

s_vs = s_samples[:, 0]

s_as = np.zeros((N,))
s_as[is_class_1] = s_samples[is_class_1, 0]
s_as[np.logical_not(is_class_1)] = s_samples[np.logical_not(is_class_1), 1]


def discretize_s(s):
    bins = np.array([-np.Inf, -9, -3, 3, 9, np.Inf])
    s_index = np.digitize(s, bins)  # 1,2,3,4,5
    s = (s_index - 3) * 6
    s_index -= 1  # 0,1,2,3,4
    return s, s_index


s_as_discrete, _ = discretize_s(s_as)
s_vs_discrete, _ = discretize_s(s_vs)


def generate_experiment_data(s_vs, s_as, params):  # TODO make this faster
    estimated_s_vs = np.zeros_like(s_vs)
    estimated_s_as = np.zeros_like(s_as)
    for i in range(estimated_s_vs.shape[0]):
        s_v, s_a = s_vs[i], s_as[i]
        x_vs = np.random.normal(s_v, params.sigma_v, 1)
        x_as = np.random.normal(s_a, params.sigma_a, 1)
        estimated_s_vs[i] = model.estimated_s_v(x_vs, x_as, params)
        estimated_s_as[i] = model.estimated_s_a(x_vs, x_as, params)

    estimated_s_vs, estimated_s_vs_index = discretize_s(estimated_s_vs)
    estimated_s_as, estimated_s_as_index = discretize_s(estimated_s_as)

    return estimated_s_vs, estimated_s_vs_index, estimated_s_as, estimated_s_as_index


estimated_s_vs, estimated_s_vs_index, estimated_s_as, estimated_s_as_index = generate_experiment_data(s_vs, s_as,
                                                                                                      true_params)


# def make_button_presses_and_visualize(N, params):
#     s_set = [-12, -6, 0, 6, 12]
#     for s_v, s_a in product(s_set, s_set):
#         x_vs = np.random.normal(s_v, params.sigma_v, N)
#         x_as = np.random.normal(s_a, params.sigma_a, N)
#         estimated_s_vs = model.estimated_s_v(x_vs, x_as, params)
#         estimated_s_as = model.estimated_s_a(x_vs, x_as, params)
#
#         hist, bin_edges = np.histogram(estimated_s_vs, bins=[-np.inf, -9, -3, 3, 9, np.inf])
#         plt.bar(s_set, hist, width = 1.5, tick_label=s_set, align='center')
#         plt.title('$\hat{{s}}_v$ for $s_v={}$ and $s_a={}$'.format(s_v, s_a))
#         plt.show()
#
#         hist, bin_edges = np.histogram(estimated_s_as, bins=[-np.inf, -9, -3, 3, 9, np.inf])
#         plt.bar(s_set, hist, width = 1.5, tick_label=s_set, align='center')
#         plt.title('$\hat{{s}}_a$ for $s_v={}$ and $s_a={}$'.format(s_v, s_a))
#         plt.show()
#
#
# make_button_presses_and_visualize(N, params)


def calculate_pmf(params):
    N = 10 ** 5
    s_set = [-12, -6, 0, 6, 12]
    s_v_s_a_to_pmf = dict()
    x_vnoise = np.random.normal(0, params.sigma_v, N)
    x_anoise = np.random.normal(0, params.sigma_a, N)
    for s_v, s_a in product(s_set, s_set):
        # a = time.time()
        x_vs = s_v + x_vnoise
        x_as = s_a + x_anoise
        # print('normaling {0:.3f}'.format(time.time() - a))

        # a = time.time()
        # most time is spent there
        estimated_s_vs, estimated_s_as = model.estimated_s_efficient(x_vs, x_as, params)
        # print('estimating {0:.3f}'.format(time.time() - a))


        # a = time.time()
        estimated_s_vs_hist, bins = np.histogram(estimated_s_vs, bins=[-np.inf, -9, -3, 3, 9, np.inf])
        estimated_s_vs_pmf = estimated_s_vs_hist / np.sum(estimated_s_vs_hist)  # probabilities need to add up to 1
        estimated_s_as_hist, _ = np.histogram(estimated_s_as, bins=[-np.inf, -9, -3, 3, 9, np.inf])
        estimated_s_as_pmf = estimated_s_as_hist / np.sum(estimated_s_as_hist)  # probabilities need to add up to 1
        # print('hist making {0:.3f}'.format(time.time() - a))


        s_v_s_a_to_pmf[s_v, s_a] = estimated_s_vs_pmf, estimated_s_as_pmf
    return s_v_s_a_to_pmf

def calculate_log_likelihood(s_vs, s_as, estimated_s_vs_index, estimated_s_as_index, params):
    # a = time.time()
    s_v_s_a_to_pmf = calculate_pmf(params)
    # print('calc pmf {0:.2f}'.format(time.time() - a))

    # a = time.time()
    likelihoods = np.zeros((s_vs.shape[0], 2))
    for i in range(s_vs.shape[0]):
        s_v, s_a = s_vs[i], s_as[i]
        estimated_s_vs_pmf, estimated_s_as_pmf = s_v_s_a_to_pmf[s_v, s_a]
        likelihoods[i, 0] = estimated_s_vs_pmf[estimated_s_vs_index[i]] + 1e-5
        likelihoods[i, 1] = estimated_s_as_pmf[estimated_s_as_index[i]] + 1e-5
    log_likelihood = np.sum(np.log(likelihoods))
    # print('log likelihood {0:.2f}'.format(time.time() - a))
    return log_likelihood



def brute_force():
    dim = 2
    p_commons = np.linspace(0,1,dim)
    sigma_vs = np.linspace(0.001, 20, dim)
    sigma_as = np.linspace(0.001, 20, dim)
    sigma_ps = np.linspace(0.001, 20, dim)

    log_likelihoods = np.zeros((dim,dim,dim,dim))

    for (i_0, p_common), (i_1, sigma_v), (i_2, sigma_a), (i_3, sigma_p) in product(enumerate(p_commons), enumerate(sigma_vs), enumerate(sigma_as), enumerate(sigma_ps)):
        print(i_0, i_1, i_2, i_3)
        params = lambda: None
        params.p_common = p_common
        params.sigma_v = sigma_v
        params.sigma_a = sigma_a
        params.sigma_p = sigma_p
        params.mu_p = 0

        a = time.time()
        log_likelihoods[i_0, i_1, i_2, i_3] = calculate_log_likelihood(s_vs_discrete, s_as_discrete, estimated_s_vs_index, estimated_s_as_index, true_params)
        print('one val {0:.2f}'.format(time.time() - a))


    best_is = np.unravel_index(log_likelihoods.argmax(), log_likelihoods.shape)
    print(p_commons[best_is[0]], sigma_vs[best_is[1]], sigma_as[best_is[2]], sigma_ps[best_is[3]])

    plt.plot(p_commons, np.apply_over_axes(np.mean, log_likelihoods, (1, 2,3)).flatten())
    plt.xlabel('p_common')
    plt.ylabel('average log likelihood')


    plt.plot(sigma_vs, np.apply_over_axes(np.mean, log_likelihoods, (0, 2,3)).flatten())
    plt.xlabel('sigma_v')
    plt.ylabel('average log likelihood')


    plt.plot(sigma_as, np.apply_over_axes(np.mean, log_likelihoods, (0, 1,3)).flatten())
    plt.xlabel('sigma_a')
    plt.ylabel('average log likelihood')


    plt.plot(sigma_ps, np.apply_over_axes(np.mean, log_likelihoods, (0,1, 2)).flatten())
    plt.xlabel('sigma_p')
    plt.ylabel('average log likelihood')

def calculate_log_likelihood_wrapper(param_arr):
    param_limits = np.array([[0, 1], [0.001, 20], [0.001, 20], [0.001, 20]])
    if not np.all(np.logical_and(param_arr > param_limits[:,0], param_arr < param_limits[:,1])):
        return -np.Inf
    params = lambda: None
    params.p_common = param_arr[0]
    params.sigma_v = param_arr[1]
    params.sigma_a = param_arr[2]
    params.sigma_p = param_arr[3]
    params.mu_p = 0
    t = calculate_log_likelihood(s_vs_discrete, s_as_discrete, estimated_s_vs_index, estimated_s_as_index, params)
    if np.isnan(t):
        return -np.Inf
    else:
        return t


def mcmc():
    ndim = 4
    # param_limits = np.array([[0, 1], [0.001, 20], [0.001, 20], [0.001, 20]])
    nwalkers = ndim**4
    start = np.random.uniform(0.001,20,(nwalkers,ndim))
    start[:,0] = np.random.uniform(0,1,(nwalkers))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, calculate_log_likelihood_wrapper)
    pos, prob, state = sampler.run_mcmc(start, 2)
    sampler.reset()
    sampler.run_mcmc(pos, 50)
    for i in range(ndim):
        plt.figure()
        plt.hist(sampler.flatchain[:,i], 500, color="k", histtype="step")
        plt.title("Dimension {0:d}".format(i))
    plt.show()
    sampler.flatchain.dump('mcmc.dump')

mcmc()

mcmc()
