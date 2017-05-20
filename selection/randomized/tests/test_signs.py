import numpy as np
from scipy.stats import norm as ndist
import pandas as pd
import selection.tests.reports as reports
from selection.tests.flags import SMALL_SAMPLES, SET_SEED
from selection.tests.decorators import wait_for_return_value, set_seed_iftrue, set_sampling_params_iftrue, register_report

from selection.distributions.discrete_family import discrete_family
import statsmodels.api as sm
from scipy.optimize import bisect


def restricted_gaussian(Z, interval=[-5.,5.]):
    L_restrict, U_restrict = interval
    Z_restrict = max(min(Z, U_restrict), L_restrict)
    return ((ndist.cdf(Z_restrict) - ndist.cdf(L_restrict)) /
            (ndist.cdf(U_restrict) - ndist.cdf(L_restrict)))

def naive_pivot(obs, sigma, truth):
    _pivot = ndist.cdf(np.true_divide(abs(obs-truth), sigma))
    _pivot = 2 * min(_pivot, 1 - _pivot)
    return _pivot

def naive_interval(obs, sigma, alpha):
    _interval = (obs - ndist.ppf(1 - alpha / 2) * sigma,
                 obs + ndist.ppf(1 - alpha / 2) * sigma)
    return _interval

def selective_pivot(obs, threshold, sigma, truth):
    F = restricted_gaussian
    #F = ndist.cdf
    obs_cs = np.true_divide(obs-truth, sigma)
    lower_cs = np.true_divide(-threshold-truth, sigma)
    upper_cs = np.true_divide(threshold-truth, sigma)

    if (F(lower_cs)-F(upper_cs))>-1:
        v = (F(min(lower_cs, obs_cs)) + F(max(obs_cs, upper_cs))-F(upper_cs))/\
             (F(lower_cs) +1- F(upper_cs))
    elif F(lower_cs) < 0.1:
        v = 1
    else:
        v = 0
    return v


def selective_interval(obs, threshold, sigma, alpha=0.1):

    lb = obs - 5 * sigma
    ub = obs + 5 * sigma

    def F(param):
        return selective_pivot(obs, threshold, sigma, truth=param)

    FL = lambda x: (F(x) - (1 - 0.5 * alpha))
    FU = lambda x: (F(x) - 0.5* alpha)
    L_conf = bisect(FL, lb, ub)
    U_conf = bisect(FU, lb, ub)
    return np.array([L_conf, U_conf])


def coverage(LU, truth):
    L, U = LU
    _length = U - L
    _covered = 0
    if (L <= truth) and (U >= truth):
        _covered = 1
    return _covered, _length

def sign_decision(obs, pvalue, pvalue_threshold):
    lb = np.true_divide(pvalue_threshold,2)
    up = 1-np.true_divide(pvalue_threshold,2)
    if (pvalue<lb) or (pvalue>up):
        return np.sign(obs)
    else:
        return None


@register_report(['pvalue', 'cover', 'ci_length_clt', 'sign_decision_selective',
                  'naive_pvalues', 'covered_naive', 'ci_length_naive', 'sign_decision_art',
                  'true_sign','active_var'])
@set_seed_iftrue(SET_SEED)
@set_sampling_params_iftrue(SMALL_SAMPLES, burnin=10, ndraw=10)
@wait_for_return_value()
def test_simple_problem(n=100, sigma=1, alpha=0.1, alpha_S=0.8):
    #truth = -1./np.sqrt(n)
    truth = -1./n
    true_sign = np.sign(truth)
    y = sigma*np.random.standard_normal(n) + truth

    obs = np.sqrt(n) * np.mean(y)
    threshold = ndist.ppf(1-(alpha/2), 0, 1)
    if abs(obs)<threshold:
        return None
    _pivot = selective_pivot(obs, threshold, sigma, np.sqrt(n)*truth)
    _pvalue = selective_pivot(obs, threshold, sigma, truth=0)
    _selective_ci = selective_interval(obs, threshold, sigma, alpha=0.1)
    _covered, _length = coverage(_selective_ci, np.sqrt(n)*truth)

    _naive_pivot = naive_pivot(obs, sigma, np.sqrt(n)*truth)
    _naive_pvalue = naive_pivot(obs, sigma, truth=0)
    _naive_ci = naive_interval(obs, sigma, alpha=0.1)
    _naive_covered, _naive_length = coverage(_naive_ci, np.sqrt(n)*truth)

    _art_sign_decision = sign_decision(obs, _naive_pvalue, 2*alpha_S*alpha)
    _selective_sign_decision = sign_decision(obs, _pvalue, alpha_S)

    print("signs", _art_sign_decision, _selective_sign_decision)

    return _pivot, _covered, _length, _selective_sign_decision, \
           _naive_pivot, _naive_covered, _naive_length, _art_sign_decision, \
           true_sign, [False]



def report(niter=100, **kwargs):

    report = reports.reports['test_simple_problem']
    results = reports.collect_multiple_runs(report['test'],
                                             report['columns'],
                                             niter,
                                             reports.summarize_all,
                                             **kwargs)

    results.to_pickle("test_simple_problem.pkl")
    results = pd.read_pickle("test_simple_problem.pkl")

    fig = reports.pivot_plot_plus_naive(results)
    fig.suptitle("Pivots simple problem", fontsize=20)
    fig.savefig('pivots_simple_problem.pdf')


if __name__ == '__main__':

    np.random.seed(500)
    kwargs = {'n': 100, 'sigma': 1}
    report(niter=1000, **kwargs)
