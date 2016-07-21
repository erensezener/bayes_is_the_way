import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import model

N = int(1e5)

params = lambda: None
params.p_common = 0.8
params.sigma_v = 0.6
params.sigma_a = 3.1
params.sigma_p = 15
params.mu_p = 0

cs = np.sign(np.random.rand(N) - params.p_common) / 2. + 1.5  # only consists of 1s and 2s
is_class_1 = np.isclose(cs, 1.)
s_samples = np.random.normal(params.mu_p, params.sigma_p, (N, 2))

s_vs = s_samples[:, 0]

s_as = np.zeros((N,))
s_as[is_class_1] = s_samples[is_class_1, 0]
s_as[np.logical_not(is_class_1)] = s_samples[np.logical_not(is_class_1), 1]

def make_button_presses(N, params):
    s_set = [-12, -6, 0, 6, 12]
    for s_v, s_a in product(s_set, s_set):
        x_vs = np.random.normal(s_v, params.sigma_v, N)
        x_as = np.random.normal(s_a, params.sigma_a, N)
        estimated_s_vs = model.estimated_s_v(x_vs, x_as, params)
        estimated_s_as = model.estimated_s_a(x_vs, x_as, params)

        hist, bin_edges = np.histogram(estimated_s_vs, bins=[-np.inf, -9, -3, 3, 9, np.inf])
        plt.bar(s_set, hist, width = 1.5, tick_label=s_set, align='center')
        plt.title('$\hat{{s}}_v$ for $s_v={}$ and $s_a={}$'.format(s_v, s_a))
        plt.show()

        hist, bin_edges = np.histogram(estimated_s_as, bins=[-np.inf, -9, -3, 3, 9, np.inf])
        plt.bar(s_set, hist, width = 1.5, tick_label=s_set, align='center')
        plt.title('$\hat{{s}}_a$ for $s_v={}$ and $s_a={}$'.format(s_v, s_a))
        plt.show()


make_button_presses(N, params)



