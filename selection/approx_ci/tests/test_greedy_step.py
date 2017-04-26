from __future__ import print_function
import numpy as np
import pandas as pd
import time
import regreg.api as rr
import selection.tests.reports as reports
from selection.tests.instance import logistic_instance, gaussian_instance
from selection.approx_ci.ci_approx_greedy_step import approximate_conditional_density
from selection.approx_ci.estimator_approx import greedy_score_step_approx

from selection.tests.flags import SMALL_SAMPLES, SET_SEED
from selection.tests.decorators import wait_for_return_value, register_report, set_sampling_params_iftrue
from selection.randomized.query import naive_confidence_intervals
from selection.randomized.query import naive_pvalues


@register_report(['truth', 'cover', 'ci_length_clt',
                  'naive_pvalues', 'covered_naive', 'ci_length_naive',
                  'active_var'])
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
@wait_for_return_value()
def test_greedy_step(n=200,
                     p=50,
                     s=0,
                     snr=5,
                     rho=0.1,
                     lam_frac = 1.,
                     loss='gaussian',
                     randomizer='gaussian'):

    from selection.api import randomization

    if loss == "gaussian":
        X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=rho, snr=snr, sigma=1.)
        loss = rr.glm.gaussian(X, y)
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
    elif loss == "logistic":
        X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=rho, snr=snr)
        loss = rr.glm.logistic(X, y)
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))

    if randomizer=='gaussian':
        randomization = randomization.isotropic_gaussian((p,), scale=1.)
    elif randomizer=='laplace':
        randomization = randomization.laplace((p,), scale=1.)

    W = np.ones(p) * lam
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    #active_bool = np.zeros(p, np.bool)
    #active_bool[range(3)] = 1
    #inactive_bool = ~active_bool

    GS = greedy_score_step_approx(loss,
                                  penalty,
                                  np.zeros(p, dtype=bool),
                                  np.ones(p, dtype=bool),
                                  randomization,
                                  randomizer)

    GS.solve_approx()
    active = GS._overall
    print("nactive", active.sum())

    ci = approximate_conditional_density(GS)
    ci.solve_approx()

    active_set = np.asarray([i for i in range(p) if active[i]])
    true_support = np.asarray([i for i in range(p) if i < s])
    nactive = np.sum(active)
    print("active set, true_support", active_set, true_support)
    true_vec = beta[active]
    #print("true coefficients", true_vec)

    active_var = np.zeros(nactive, np.bool)
    for i in range(nactive):
        active_var[i] = active_set[i] in true_support

    if (set(active_set).intersection(set(true_support)) == set(true_support)) == True:

        ci = approximate_conditional_density(GS)
        ci.solve_approx()
        pvalues = ci.approximate_pvalues()
        if pvalues is None:
            return None

        class target_class(object):
            def __init__(self, target_cov):
                self.target_cov = target_cov
                self.shape = target_cov.shape

        target = target_class(GS.target_cov)
        naive_pvals = naive_pvalues(target, GS.target_observed, true_vec)

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

        naive_ci = naive_confidence_intervals(target, GS.target_observed)
        naive_covered, naive_length = coverage(naive_ci)

        return pvalues, sel_covered, sel_length, \
               naive_pvals, naive_covered, naive_length, \
               active_var


def report(niter=200, **kwargs):


    split_report = reports.reports['test_greedy_step']
    results = reports.collect_multiple_runs(split_report['test'],
                                            split_report['columns'],
                                            niter,
                                            reports.summarize_all,
                                            **kwargs)

    pdf_label = "_".join(["alternative_bootstrap_gs", kwargs['loss'], str(kwargs['n']), str(kwargs['p']), ".pdf"])
    pkl_label = "_".join(["alternative_bootstrap_gs", kwargs['loss'], str(kwargs['n']), str(kwargs['p']), ".pkl"])

    results.to_pickle(pkl_label)
    read_results = pd.read_pickle(pkl_label)
    fig = reports.pivot_plot_plus_naive(read_results)
    fig.suptitle("Alternative bootstrap GS", fontsize=20)
    fig.savefig(pdf_label)


if __name__=='__main__':
    kwargs = {'s': 0, 'n': 200, 'p': 30, 'snr': 7, 'loss': 'gaussian', 'randomizer': 'gaussian'}
    report(niter = 50, **kwargs)
