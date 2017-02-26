from __future__ import print_function
import numpy as np

import regreg.api as rr

from selection.tests.decorators import wait_for_return_value, register_report, set_sampling_params_iftrue
import selection.tests.reports as reports
from selection.tests.flags import SMALL_SAMPLES

from selection.api import multiple_queries, loco_target
from selection.randomized.glm import split_glm_group_lasso
from selection.tests.instance import (gaussian_instance, logistic_instance)

from selection.randomized.query import (naive_pvalues, naive_confidence_intervals)

@register_report(['mle', 'truth', 'pvalue', 'cover', 'ci_length_clt',
                  'naive_pvalues', 'naive_cover','ci_length_naive', 'active'])
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
@wait_for_return_value()
def test_loco(s=3,
              n=200,
              p=50,
              snr=7,
              rho=0.,
              split_frac=0.8,
              lam_frac=0.7,
              loss_label = 'gaussian',
              ndraw=10000,
              burnin=2000,
              bootstrap=False,
              solve_args={'min_its':50, 'tol':1.e-10},
              reference_known=False):

    if loss_label == "gaussian":
        X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=rho, snr=snr, sigma=1)
        lam = np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 1000))))) * sigma
        loss = rr.glm.gaussian(X, y)
        loss_rr = rr.glm.gaussian
    elif loss_label == "logistic":
        X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=rho, snr=snr)
        loss = rr.glm.logistic(X, y)
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))
        loss_rr = rr.glm.logistic

    nonzero = np.where(beta)[0]

    m = int(split_frac * n)
    epsilon = 1. / np.sqrt(n)

    W = np.ones(p)*lam
    W[0] = 0 # use at least some unpenalized
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    M_est = split_glm_group_lasso(loss, epsilon, m, penalty)
    mv = multiple_queries([M_est])
    mv.solve()

    active = M_est.selection_variable['variables']
    true_vec = beta[active]
    nactive = np.sum(active)

    idx = M_est.randomization.idx
    loss_1 = loss_rr(X[idx,:], y[idx])
    loss_2 = loss_rr(X[~idx,:], y[~idx])

    if nactive==0:
        return None

    if set(nonzero).issubset(np.nonzero(M_est.selection_variable['variables'])[0]):

        active_set = np.nonzero(active)[0]

        target_sampler, target_observed = loco_target(loss_1, loss_2,
                                                      glm_loss=loss_rr,
                                                      active=active,
                                                      epsilon=0.2,
                                                      lam=lam,
                                                      queries=mv,
                                                      bootstrap=bootstrap)

        target_sample = target_sampler.sample(ndraw=ndraw,
                                              burnin=burnin)

        LU = target_sampler.confidence_intervals(target_observed,
                                                 sample=target_sample).T

        LU_naive = naive_confidence_intervals(target_sampler, target_observed)

        pivots_mle = target_sampler.coefficient_pvalues(target_observed,
                                                        parameter=target_sampler.reference,
                                                        sample=target_sample)

        true_vec = target_observed # needs to be changed
        pivots_truth = target_sampler.coefficient_pvalues(target_observed,
                                                          parameter=true_vec,
                                                          sample=target_sample)

        pvalues = target_sampler.coefficient_pvalues(target_observed,
                                                     parameter=np.zeros_like(true_vec),
                                                     sample=target_sample)

        L, U = LU

        covered = np.zeros(nactive, np.bool)
        ci_length_sel = np.zeros(nactive)
        naive_covered = np.zeros(nactive, np.bool)
        active_var = np.zeros(nactive, np.bool)
        ci_length_naive = np.zeros(nactive)

        for j in range(nactive):
            if (L[j] <= true_vec[j]) and (U[j] >= true_vec[j]):
                covered[j] = 1
            ci_length_sel[j] = U[j]-L[j]
            if (LU_naive[j,0] <= true_vec[j]) and (LU_naive[j,1] >= true_vec[j]):
                naive_covered[j] = 1
            ci_length_naive[j] = LU_naive[j, 1] - LU_naive[j, 0]

            active_var[j] = active_set[j] in nonzero

        naive_pvals = naive_pvalues(target_sampler, target_observed, true_vec)

        return pivots_mle, pivots_truth, pvalues, covered, ci_length_sel,\
               naive_pvals, naive_covered, ci_length_naive, active_var


def report(niter=50, **kwargs):

    split_report = reports.reports['test_loco']
    runs = reports.collect_multiple_runs(split_report['test'],
                                             split_report['columns'],
                                             niter,
                                             reports.summarize_all,
                                             **kwargs)
    fig = reports.pivot_plot_plus_naive(runs)
    fig.suptitle('Randomized Lasso marginalized subgradient')
    fig.savefig('marginalized_subgrad_pivots.pdf')


if __name__== '__main__':
    report()
