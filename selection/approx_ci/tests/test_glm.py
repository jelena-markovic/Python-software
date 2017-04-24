from __future__ import print_function
import numpy as np
import pandas as pd
import regreg.api as rr
import selection.tests.reports as reports
from selection.tests.instance import logistic_instance, gaussian_instance
from selection.approx_ci.ci_via_approx_density import approximate_conditional_density
from selection.approx_ci.estimator_approx import M_estimator_approx

from selection.tests.flags import SMALL_SAMPLES, SET_SEED
from selection.tests.decorators import wait_for_return_value, register_report, set_sampling_params_iftrue
from selection.randomized.query import naive_confidence_intervals
from selection.randomized.query import (naive_pvalues, naive_confidence_intervals)


@register_report(['truth', 'cover', 'ci_length_clt',
                  'naive_pvalues', 'covered_naive', 'ci_length_naive',
                  'active_var'])
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
@wait_for_return_value()
def test_glm(n=500,
             p=20,
             s=0,
             snr=3,
             rho=0.,
             lam_frac = 1.,
             loss='gaussian',
             randomizer='gaussian'):

    from selection.api import randomization
    if snr == 0:
        s=0
    if loss == "gaussian":
        X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=rho, snr=snr, sigma=1.)
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
        loss = rr.glm.gaussian(X, y)
    elif loss == "logistic":
        X, y, beta, nonzero = logistic_instance(n=n, p=p, s=s, rho=rho, snr=snr)
        loss = rr.glm.logistic(X, y)
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))

    true_support = nonzero
    epsilon = 1. / np.sqrt(n)

    W = np.ones(p) * lam
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)
    if randomizer=='gaussian':
        randomization = randomization.isotropic_gaussian((p,), scale=1.)
    elif randomizer=='laplace':
        randomization = randomization.laplace((p,), scale=1.)

    M_est = M_estimator_approx(loss, epsilon, penalty, randomization, randomizer)
    M_est.solve_approx()
    active = M_est._overall
    active_set = np.asarray([i for i in range(p) if active[i]])
    nactive = np.sum(active)
    if nactive==0:
        return None
    print("active set, true_support", active_set, true_support)
    true_vec = beta[active]

    active_var = np.zeros(nactive, np.bool)
    for i in range(nactive):
        active_var[i] = active_set[i] in true_support
    #print("true coefficients", true_vec)

    if (set(active_set).intersection(set(true_support)) == set(true_support)) == True:
        ci = approximate_conditional_density(M_est)
        ci.solve_approx()
        pvalues = ci.approximate_pvalues()
        if pvalues is None:
            return None


        class target_class(object):
            def __init__(self, target_cov):
                self.target_cov = target_cov
                self.shape = target_cov.shape

        target = target_class(M_est.target_cov)
        naive_pvals = naive_pvalues(target, M_est.target_observed, true_vec)

        def coverage(LU):
            L, U = LU[:, 0], LU[:, 1]
            covered = np.zeros(nactive)
            ci_length = np.zeros(nactive)

            for j in range(nactive):
                if (L[j] <= true_vec[j]) and (U[j] >= true_vec[j]):
                        covered[j] = 1
                ci_length[j] = U[j] - L[j]
            return covered, ci_length

        selective_ci = ci.approximate_confidence_intervals()
        if selective_ci is None:
            return None
        sel_covered, sel_length = coverage(selective_ci)

        naive_ci = naive_confidence_intervals(target, M_est.target_observed)
        naive_covered, naive_length = coverage(naive_ci)

        return pvalues, sel_covered, sel_length, \
               naive_pvals, naive_covered, naive_length, \
               active_var
    #else:
    #    return 0

def report(niter=50, **kwargs):

    kwargs = {'s': 0, 'n': 500, 'p': 100, 'snr': 5, 'loss': 'gaussian', 'randomizer':'gaussian'}
    split_report = reports.reports['test_glm']
    results = reports.collect_multiple_runs(split_report['test'],
                                                     split_report['columns'],
                                                     niter,
                                                     reports.summarize_all,
                                                     **kwargs)
    results.to_pickle("alt_bootstrap.pkl")
    read_results = pd.read_pickle("alt_bootstrap.pkl")
    fig = reports.pivot_plot_plus_naive(read_results)
    fig.suptitle("Alternative bootstrap", fontsize=20)
    fig.savefig('alt_bootstrap.pdf')


if __name__=='__main__':

    report()

