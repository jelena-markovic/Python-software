from __future__ import print_function
import numpy as np
import pandas as pd
import regreg.api as rr
import selection.tests.reports as reports


from selection.tests.flags import SET_SEED, SMALL_SAMPLES
from selection.tests.instance import logistic_instance, gaussian_instance
from selection.tests.decorators import (wait_for_return_value,
                                        set_seed_iftrue,
                                        set_sampling_params_iftrue,
                                        register_report)
import selection.tests.reports as reports

from selection.api import (randomization,
                           glm_group_lasso,
                           pairs_bootstrap_glm,
                           multiple_queries,
                           glm_target,
                           glm_group_lasso_parametric)

from selection.randomized.query import (naive_confidence_intervals, naive_pvalues)

from selection.randomized.glm import standard_ci

@register_report(['truth', 'covered_clt', 'ci_length_clt',
                  'naive_pvalues', 'covered_naive', 'ci_length_naive',
                  'split_pvalues', 'covered_split', 'ci_length_split'])
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
@set_seed_iftrue(SET_SEED)
@wait_for_return_value()
def test_new_data(s=0,
                  n=300,
                  p=50,
                  rho=0.,
                  signal=3.5,
                  lam_frac = 1.,
                  ndraw=10000,
                  burnin=2000,
                  loss='gaussian',
                  randomizer ='laplace',
                  randomizer_scale =1.,
                  scalings=False,
                  subgrad =True,
                  check_screen = False):

    if loss=="gaussian":
        X1, y1, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=rho, signal=signal, sigma=1, random_signs=False)
        lam = lam_frac * np.mean(np.fabs(np.dot(X1.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
        loss1 = rr.glm.gaussian(X1, y1)
        X2, y2, _, _, _ = gaussian_instance(n=n, p=p, s=s, rho=rho, signal=signal, sigma=1, random_signs=False)
        loss2 = rr.glm.gaussian(X2, y2)
        loss = rr.glm.gaussian(np.concatenate((X1, X2), axis=0), np.concatenate((y1,y2), axis=0))
        X, y = loss.data
        rr_loss = rr.glm.gaussian
    elif loss=="logistic":
        X1, y1, beta, _ = logistic_instance(n=n, p=p, s=s, rho=rho, signal=signal, random_signs=False)
        loss1 = rr.glm.logistic(X1, y1)
        lam = lam_frac * np.mean(np.fabs(np.dot(X1.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))
        X2, y2, _, _ = logistic_instance(n=n, p=p, s=s, rho=rho, signal=signal, random_signs=False)
        loss2 = rr.glm.logistic(X2, y2)
        loss = rr.glm.logistic(np.concatenate((X1, X2), axis=0), np.concatenate((y1,y2), axis=0))
        rr_loss = rr.glm.logistic
    nonzero = np.where(beta)[0]

    if randomizer == 'laplace':
        randomizer = randomization.laplace((p,), scale=randomizer_scale)
    elif randomizer == 'gaussian':
        randomizer = randomization.isotropic_gaussian((p,), scale=randomizer_scale)

    epsilon = 1./np.sqrt(n)
    W = np.ones(p)*lam
    #W[0] = 0 # use at least some unpenalized
    penalty = rr.group_lasso(np.arange(p), weights=dict(zip(np.arange(p), W)), lagrange=1.)
    M_est = glm_group_lasso(loss1, epsilon, penalty, randomizer)
    M_est.solve()
    views = [M_est]
    queries = multiple_queries(views)
    queries.solve()


    active_union = M_est._overall
    nactive = np.sum(active_union)
    print("nactive", nactive)
    active_set = np.nonzero(active_union)[0]
    print("active set", active_set)
    print("true nonzero", np.nonzero(beta)[0])
    true_vec = beta[active_union]


    screened = False
    if set(nonzero).issubset(np.nonzero(active_union)[0]):
        screened = True

    if check_screen==False or (check_screen==True and screened==True):

        #if nactive==s:
        #    return None

        if scalings: # try condition on some scalings
            M_est.condition_on_subgradient()
            M_est.condition_on_scalings()
        if subgrad:
            M_est.decompose_subgradient(conditioning_groups=np.zeros(p, dtype=bool), marginalizing_groups=np.ones(p, bool))


        target_sampler, target_observed = glm_target(loss,
                                                     active_union,
                                                     queries,
                                                     bootstrap=False)

        target_sample = target_sampler.sample(ndraw=ndraw,
                                              burnin=burnin)
        LU = target_sampler.confidence_intervals(target_observed,
                                                 sample=target_sample,
                                                 level=0.9)
        pivots = target_sampler.coefficient_pvalues(target_observed,
                                                    parameter=np.zeros(nactive),
                                                    sample=target_sample)

        def coverage(LU):
            L, U = LU[:, 0], LU[:, 1]
            covered = np.zeros(nactive)
            ci_length = np.zeros(nactive)
            for j in range(nactive):
                if (L[j] <= true_vec[j]) and (U[j] >= true_vec[j]):
                        covered[j] = 1
                ci_length[j] = U[j] - L[j]
            return covered, ci_length

        covered, ci_length = coverage(LU)
        LU_naive = naive_confidence_intervals(target_sampler, target_observed)
        covered_naive, ci_length_naive = coverage(LU_naive)
        naive_pvals = naive_pvalues(target_sampler, target_observed, np.zeros(nactive))

        LU_split, split_pvalues = standard_ci(rr_loss, X2, y2, active_union, np.ones(n, np.bool))
        covered_split, ci_length_split = coverage(LU_split)

        return pivots, covered, ci_length, \
               naive_pvals, covered_naive, ci_length_naive, \
               split_pvalues, covered_split, ci_length_split


def report(niter=1, **kwargs):

    condition_report = reports.reports['test_new_data']
    runs = reports.collect_multiple_runs(condition_report['test'],
                                         condition_report['columns'],
                                         niter,
                                         reports.summarize_all,
                                         **kwargs)

    label = "test_new_data"

    pkl_label = ''.join([label, ".pkl"])
    pdf_label = ''.join([label, ".pdf"])
    runs.to_pickle(pkl_label)
    runs_read = pd.read_pickle(pkl_label)

    fig = reports.pivot_plot_plus_naive(runs_read, color='b', label='no screening')

    fig.suptitle('Testing without screening', fontsize=20)
    fig.savefig(pdf_label)


if __name__ == '__main__':
    np.random.seed(500)
    #kwargs = {'s':30, 'n':3000, 'p':1000, 'signal':3.5, 'rho':0, 'loss':'gaussian', 'randomizer':'gaussian',
    #              'randomizer_scale':1.2, 'lam_frac':1.}
    #report(niter=1, **kwargs)
    report(niter=50)