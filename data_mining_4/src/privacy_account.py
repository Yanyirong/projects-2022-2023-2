import numpy as np
from scipy import optimize
from scipy.stats import norm

import math

"""
Optionally you could use moments accountant to implement the epsilon calculation.
"""


# def delta_eps_mu(eps, mu):
#     return norm.cdf(-eps / mu + mu / 2) - np.exp(eps) * norm.cdf(-eps / mu - mu / 2)

def get_epsilon(epoch, delta, sigma, sensitivity, batch_size, training_nums):
    """
    Compute epsilon with basic composition from given epoch, delta, sigma, sensitivity, batch_size and the number of training set.
    """
    q = batch_size / training_nums
    steps = int(math.ceil(training_nums / batch_size * epoch))
    mu = np.sqrt(np.exp(sigma ** (-2)) - 1) * np.sqrt(steps) * q
    
    def f(x):
        eps = x
        ans = norm.cdf(-eps / mu + mu / 2) - np.exp(eps) * norm.cdf(-eps / mu - mu / 2) - delta
        return ans
    
    return optimize.root_scalar(f, bracket=[0, 500], method="brentq").root