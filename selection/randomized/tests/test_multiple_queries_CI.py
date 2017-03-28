from __future__ import print_function
import numpy as np

import regreg.api as rr

import selection.tests.reports as reports
import pandas as pd

from selection.tests.flags import SMALL_SAMPLES, SET_SEED
from selection.api import (randomization,
                           glm_group_lasso,
                           multiple_queries,
                           glm_target)
from selection.tests.instance import logistic_instance, gaussian_instance
from selection.tests.decorators import wait_for_return_value, register_report, set_sampling_params_iftrue
from selection.randomized.query import naive_confidence_intervals, naive_pvalues

@register_report(['pivots_clt', 'pivots_boot',
                  'covered_clt', 'ci_length_clt',
                  'covered_boot', 'ci_length_boot',
                  'active',
                   'pivots_naive','covered_naive', 'ci_length_naive'])
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
@wait_for_return_value()
def test_multiple_queries(s=3,
                         n=300,
                         p=20,
                         snr=7,
                         rho=0.1,
                         lam_frac=0.7,
                         nviews = 4,
                         intervals ='old',
                         loss ='logistic',
                         randomizer = 'laplace',
                         randomizer_scale = 1.,
                         ndraw=10000, burnin=2000,
                         solve_args={'min_its':50, 'tol':1.e-10},
                         check_screen =True,
                         subgrad=True):
    if loss == "gaussian":
        X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=rho, snr=snr, sigma=1)
        lam = np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 1000))))) * sigma
        loss = rr.glm.gaussian(X, y)
    elif loss == "logistic":
        X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=rho, snr=snr)
        loss = rr.glm.logistic(X, y)
        lam = np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))

    if randomizer == 'laplace':
        randomizer = randomization.laplace((p,), scale=randomizer_scale)
    elif randomizer == 'gaussian':
        randomizer = randomization.isotropic_gaussian((p,), randomizer_scale)
    elif randomizer == 'logistic':
        randomizer = randomization.logistic((p,), scale=randomizer_scale)

    nonzero = np.where(beta)[0]
    epsilon = 1./np.sqrt(n)

    W = np.ones(p)*lam *lam_frac
    #W[0] = 0 # use at least some unpenalized
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    view = []
    for i in range(nviews):
        view.append(glm_group_lasso(loss, epsilon, penalty, randomizer))

    mv = multiple_queries(view)
    mv.solve()

    active_union = np.zeros(p, np.bool)
    for i in range(nviews):
        active_union += view[i].selection_variable['variables']

    nactive = np.sum(active_union)
    print("nactive", nactive)
    if nactive==0:
        return None

    screen = set(nonzero).issubset(np.nonzero(active_union)[0])

    if check_screen and not screen:
        return None

    if True:
        active_set = np.nonzero(active_union)[0]
        true_vec = beta[active_union]

        if subgrad:
            for i in range(nviews):
               view[i].decompose_subgradient(conditioning_groups=np.zeros(p, dtype=bool), marginalizing_groups=np.ones(p, bool))

        ## bootstrap
        target_sampler_boot, target_observed = glm_target(loss,
                                                          active_union,
                                                          mv,
                                                          bootstrap=True)

        if intervals == 'old':
            target_sample_boot = target_sampler_boot.sample(ndraw=ndraw,
                                                            burnin=burnin)
            LU_boot = target_sampler_boot.confidence_intervals(target_observed,
                                                               sample=target_sample_boot,
                                                               level=0.9)
            pivots_boot = target_sampler_boot.coefficient_pvalues(target_observed,
                                                                  parameter=true_vec,
                                                                  sample=target_sample_boot)
        else:
            full_sample_boot = target_sampler_boot.sample(ndraw=ndraw,
                                                          burnin=burnin,
                                                          keep_opt=True)
            LU_boot = target_sampler_boot.confidence_intervals_translate(target_observed,
                                                                         sample=full_sample_boot,
                                                                         level=0.9)
            pivots_boot = target_sampler_boot.coefficient_pvalues_translate(target_observed,
                                                                            parameter=true_vec,
                                                                            sample=full_sample_boot)
        ## CLT plugin
        target_sampler, _ = glm_target(loss,
                                       active_union,
                                       mv,
                                       bootstrap=False)

        if intervals == 'old':
            target_sample = target_sampler.sample(ndraw=ndraw,
                                                  burnin=burnin)
            LU = target_sampler.confidence_intervals(target_observed,
                                                     sample=target_sample,
                                                     level=0.9)
            pivots = target_sampler.coefficient_pvalues(target_observed,
                                                        parameter=true_vec,
                                                        sample=target_sample)
        else:
            full_sample = target_sampler.sample(ndraw=ndraw,
                                                burnin=burnin,
                                                keep_opt=True)
            LU = target_sampler.confidence_intervals_translate(target_observed,
                                                               sample=full_sample,
                                                               level=0.9)
            pivots = target_sampler.coefficient_pvalues_translate(target_observed,
                                                                  parameter=true_vec,
                                                                  sample=full_sample)

        LU_naive = naive_confidence_intervals(target_sampler, target_observed)
        naive_pivots = naive_pvalues(target_sampler, target_observed, true_vec)


        def coverage(LU):
            L, U = LU[:,0], LU[:,1]
            covered = np.zeros(nactive)
            ci_length = np.zeros(nactive)

            for j in range(nactive):
                if check_screen:
                  if (L[j] <= true_vec[j]) and (U[j] >= true_vec[j]):
                    covered[j] = 1
                else:
                    covered[j] = None
                ci_length[j] = U[j]-L[j]
            return covered, ci_length

        covered, ci_length = coverage(LU)
        covered_boot, ci_length_boot = coverage(LU_boot)
        covered_naive, ci_length_naive = coverage(LU_naive)

        active_var = np.zeros(nactive, np.bool)
        for j in range(nactive):
            active_var[j] = active_set[j] in nonzero

        return pivots, pivots_boot, covered, ci_length, covered_boot, ci_length_boot, \
                active_var, naive_pivots, covered_naive, ci_length_naive


def report(niter=1, **kwargs):

    kwargs = {'s': 0, 'n': 300, 'p': 50, 'rho': 0., 'snr': 7,
              'nviews': 3, 'intervals': 'old', 'lam_frac': 1.5,
              'randomizer': "laplace", 'loss': "logistic"}

    split_report = reports.reports['test_multiple_queries']
    screened_results = reports.collect_multiple_runs(split_report['test'],
                                                     split_report['columns'],
                                                     niter,
                                                     reports.summarize_all,
                                                     **kwargs)

    screened_results.to_pickle("multiple_queries.pkl")
    results = pd.read_pickle("multiple_queries.pkl")

    fig = reports.boot_clt_plot(results)
    fig.suptitle("Multiple queries on the data", fontsize=20)
    fig.savefig('multiple_queries.pdf')  # will have both bootstrap and CLT on plot


if __name__=='__main__':
    report()
