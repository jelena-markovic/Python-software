from __future__ import print_function
import numpy as np
import pandas as pd
import regreg.api as rr

from selection.tests.decorators import wait_for_return_value, register_report, set_sampling_params_iftrue
import selection.tests.reports as reports
from selection.tests.flags import SMALL_SAMPLES

from selection.api import multiple_queries
from selection.randomized.loco import LOCO, loco_target
from selection.randomized.glm import split_glm_group_lasso
from selection.tests.instance import (gaussian_instance, logistic_instance)

from selection.randomized.query import (naive_pvalues, naive_confidence_intervals)

@register_report(['truth', 'pvalue', 'cover', 'ci_length_clt',
                  'naive_pvalues', 'naive_cover','ci_length_naive',
                  'split_pvalues', 'covered_split', 'ci_length_split',
                  'active'])
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
@wait_for_return_value()
def test_loco(s=0,
              n=200,
              p=50,
              snr=7,
              rho=0.,
              split_frac = 0.8,
              lam_frac = 1.2,
              loss_label = 'gaussian',
              ndraw = 10000,
              burnin = 2000,
              bootstrap = False,
              solve_args = {'min_its':50, 'tol':1.e-10},
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

    W = np.ones(p)*lam_frac*lam
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
    print("nactive", nactive)
    if set(nonzero).issubset(np.nonzero(M_est.selection_variable['variables'])[0]):

        active_set = np.nonzero(active)[0]

        _loco = LOCO(loss, loss_1, loss_2,
                     loss_rr=loss_rr,
                     active=active,
                     epsilon=1., # the randomization added \espilon*unif
                     lam=lam)

        target_sampler, target_observed = loco_target(loss,
                                                      _loco.pairs_bootstrap_loco(),
                                                      queries=mv,
                                                      bootstrap=bootstrap)
                                                      # reference=np.zeros(nactive))

        target_sample = target_sampler.sample(ndraw=ndraw,
                                              burnin=burnin)

        LU = target_sampler.confidence_intervals(target_observed,
                                                 sample=target_sample).T

        LU_naive = naive_confidence_intervals(target_sampler, target_observed)


        true_vec = np.zeros_like(target_observed) # needs to be changed
        pivots_truth = target_sampler.coefficient_pvalues(target_observed,
                                                          parameter=true_vec,
                                                          sample=target_sample)

        pvalues = target_sampler.coefficient_pvalues(target_observed,
                                                     parameter=np.zeros_like(true_vec),
                                                     sample=target_sample)

        LU_split, split_pvalues = _loco.split_intervals()
        active_var = np.zeros(nactive, np.bool)


        def coverage(LU):
            L, U = LU[:, 0], LU[:, 1]
            covered = np.zeros(nactive, np.bool)
            ci_length = np.zeros(nactive)

            for j in range(nactive):
                if (L[j] <= true_vec[j]) and (U[j] >= true_vec[j]):
                        covered[j] = 1
                ci_length[j] = U[j] - L[j]
            return covered, ci_length

        for j in range(nactive):
            active_var[j] = active_set[j] in nonzero

        covered, ci_length_sel = coverage(LU.T)
        naive_covered, ci_length_naive = coverage(LU_naive)
        split_covered, ci_length_split = coverage(LU_split)

        naive_pvals = naive_pvalues(target_sampler, target_observed, true_vec)

        return pivots_truth, pvalues, covered, ci_length_sel,\
               naive_pvals, naive_covered, ci_length_naive, \
               split_pvalues, split_covered, ci_length_split, \
               active_var


def report(niter=1, **kwargs):

    split_report = reports.reports['test_loco']
    runs = reports.collect_multiple_runs(split_report['test'],
                                             split_report['columns'],
                                             niter,
                                             reports.summarize_all,
                                             **kwargs)
    runs.to_pickle("loco.pkl")
    read_runs = pd.read_pickle("loco.pkl")

    fig = reports.split_pvalue_plot(read_runs)
    fig.suptitle('Loco alternative', fontsize=20)
    fig.savefig('Loco.pdf')


if __name__== '__main__':

    kwargs = {'s': 0, 'n': 200, 'p': 50, 'snr': 7, 'rho': 0.,
              'split_frac': 0.8, 'lam_frac': 1.2,
              'loss_label': 'gaussian',
              'reference_known': False}

    report(niter=50, **kwargs)
