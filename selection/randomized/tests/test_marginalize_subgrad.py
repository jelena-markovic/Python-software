from __future__ import print_function
import numpy as np
import functools
import regreg.api as rr
import selection.tests.reports as reports
import timeit

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
                           discrete_family,
                           projected_langevin,
                           glm_group_lasso_parametric,
                           glm_target)

from selection.randomized.query import (naive_pvalues, naive_confidence_intervals)
from selection.randomized.glm import glm_parametric_covariance, glm_nonparametric_bootstrap, restricted_Mest, set_alpha_matrix
import pandas as pd
import sys
import os


def coverage(LU, true_vec, check_screen):
    nactive = true_vec.shape[0]
    L, U = LU[:, 0], LU[:, 1]
    covered = np.zeros(nactive)
    ci_length = np.zeros(nactive)

    for j in range(nactive):
        if check_screen:
            if (L[j] <= true_vec[j]) and (U[j] >= true_vec[j]):
                covered[j] = 1
        else:
            covered[j]=None
        ci_length[j] = U[j] - L[j]
    return covered, ci_length

def compute_true_vec(get_data_easy, active, nsim=2000):
    _beta_unpenalized = np.zeros(np.sum(active))
    for i in range(nsim):
        glm_loss = get_data_easy()
        #print(restricted_Mest(glm_loss, active))
        _beta_unpenalized += np.array(restricted_Mest(glm_loss, active))

    return np.true_divide(_beta_unpenalized, nsim)


def get_data(loss_label, n, p, s, rho, signal, sigma, X):
    if loss_label == "gaussian":
        X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=rho, signal=signal, sigma=sigma, X=X)
        glm_loss = rr.glm.gaussian(X, y)
    elif loss_label == "logistic":
        X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=rho, signal=signal, X=X)
        glm_loss = rr.glm.logistic(X, y)
    return glm_loss

@register_report(['truth', 'covered_clt', 'ci_length_clt',
                  'naive_pvalues', 'covered_naive', 'ci_length_naive'])
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
@set_seed_iftrue(SET_SEED)
@wait_for_return_value()
def test_marginalize(s=0,
                    n=100,
                    p=20,
                    rho=0.,
                    signal=5.,
                    sigma=1.,
                    lam_frac = 1.,
                    X = None,
                    ndraw=10000,
                    burnin=1000,
                    loss_label='gaussian',
                    randomizer = 'gaussian',
                    randomizer_scale = 1.,
                    nviews=1,
                    scalings=False,
                    subgrad =True,
                    parametric=True,
                    intervals='old',
                    check_screen=True):
    print(n,p,s)

    if randomizer == 'laplace':
        randomizer = randomization.laplace((p,), scale=randomizer_scale)
    elif randomizer == 'gaussian':
        randomizer = randomization.isotropic_gaussian((p,), randomizer_scale)
    elif randomizer == 'logistic':
        randomizer = randomization.logistic((p,), scale=randomizer_scale)

    if loss_label=="gaussian":
        X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=rho, signal=signal, sigma=1, X=X)
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 1000)))).max(0)) * sigma
        glm_loss = rr.glm.gaussian(X,y)
        print(lam)
    elif loss_label=="logistic":
        X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=rho, signal=signal, X= X)
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 1000)))).max(0))
        glm_loss = rr.glm.logistic(X,y)

    epsilon = 1. / np.sqrt(n)
    W = np.ones(p)*lam
    #W[0] = 0 # use at least some unpenalized
    penalty = rr.group_lasso(np.arange(p), weights=dict(zip(np.arange(p), W)), lagrange=1.)

    views = []
    for i in range(nviews):
        if parametric==False:
            views.append(glm_group_lasso(glm_loss, epsilon, penalty, randomizer))
        else:
            views.append(glm_group_lasso_parametric(glm_loss, epsilon, penalty, randomizer))

    queries = multiple_queries(views)
    queries.solve()

    active_union = np.zeros(p, np.bool)
    for view in views:
        active_union += view.selection_variable['variables']

    nactive = np.sum(active_union)
    print("nactive", nactive)
    if nactive==0:
        return None
    active_set = np.nonzero(active_union)[0]
    print("active set", active_set)
    nonzero = np.where(beta)[0]

    get_data_easy = functools.partial(get_data, loss_label, n, p, s, rho, signal, sigma, X)
    #start = start = timeit.default_timer()
    #true_vec = compute_true_vec(get_data_easy, active_union)
    #stop = timeit.default_timer()
    #print("computing true vec time", stop - start)
    true_vec = beta[active_union]

    if set(nonzero).issubset(active_set) or check_screen==False:

        if scalings: # try condition on some scalings
            for i in range(nviews):
                views[i].condition_on_subgradient()
                views[i].condition_on_scalings()
        if subgrad:
            for i in range(nviews):
               views[i].decompose_subgradient(conditioning_groups=np.zeros(p, dtype=bool), marginalizing_groups=np.ones(p, bool))

        start = timeit.default_timer()
        target_sampler, target_observed = glm_target(glm_loss,
                                                     active_union,
                                                     queries,
                                                     bootstrap=False,
                                                     parametric=parametric,
                                                     opt=True)
                                                     #reference= beta[active_union])
        stop = timeit.default_timer()
        print("target setup time", stop-start)

        if intervals=='old':
            start = timeit.default_timer()
            target_sample = target_sampler.sample(ndraw=ndraw,
                                                  burnin=burnin)
            print(target_sample.shape)
            stop = timeit.default_timer()
            print("sampling time", stop - start)
            start = timeit.default_timer()
            pivots = target_sampler.coefficient_pvalues(target_observed,
                                                        parameter=true_vec,
                                                        sample=target_sample)
            stop = timeit.default_timer()
            print("pivots time", stop - start)
            start = timeit.default_timer()
            LU = target_sampler.confidence_intervals(target_observed,
                                                     sample=target_sample,
                                                     level=0.9)
            stop = timeit.default_timer()
            print("confidence intervals time", stop - start)
        elif intervals=='new':
            full_sample = target_sampler.sample(ndraw=ndraw,
                                                burnin=burnin,
                                                keep_opt=True)
            LU = target_sampler.confidence_intervals_translate(target_observed,
                                                           sample=full_sample,
                                                           level=0.9)
            pivots = target_sampler.coefficient_pvalues_translate(target_observed,
                                                                    parameter=true_vec,
                                                                    sample=full_sample)

        #test_stat = lambda x: np.linalg.norm(x - beta[active_union])
        #observed_test_value = test_stat(target_observed)
        #pivots = target_sampler.hypothesis_test(test_stat,
        #                                       observed_test_value,
        #                                       alternative='twosided',
        #                                       parameter = beta[active_union],
        #                                       ndraw=ndraw,
        #                                       burnin=burnin,
        #                                       stepsize=None)

        covered, ci_length = coverage(LU, true_vec, check_screen)
        LU_naive = naive_confidence_intervals(target_sampler, target_observed)
        covered_naive, ci_length_naive = coverage(LU_naive, true_vec, check_screen)
        naive_pvals = naive_pvalues(target_sampler, target_observed, true_vec)

        return pivots, covered, ci_length, naive_pvals, covered_naive, ci_length_naive


def report(niter, outfile, **kwargs):
    rho = 0
    n = 100
    p = 20
    X = (np.sqrt(1 - rho) * np.random.standard_normal((n, p)) +
                   np.sqrt(rho) * np.random.standard_normal(n)[:, None])

    kwargs = {'X':None, 'rho':rho, 'n':n, 'p':p}
    condition_report = reports.reports['test_marginalize']
    runs = reports.collect_multiple_runs(condition_report['test'],
                                         condition_report['columns'],
                                         niter,
                                         reports.summarize_all,
                                         **kwargs)
    if outfile is None:
        outfile = "marginalize_subgrad.pkl"
    runs.to_pickle(outfile)

    #results = pd.read_pickle(outfile)
    #fig = reports.pivot_plot_plus_naive(results)
    #fig.suptitle('Randomized Lasso marginalized subgradient')
    #fig.savefig('marginalized_subgrad_pivots.pdf')


if __name__ == '__main__':
    cluster = False
    if cluster==True:
        seedn = int(sys.argv[1])
        outdir = sys.argv[2]
        outfile = os.path.join(outdir, "list_result_" + str(seedn) + ".pkl")
    else:
        outfile=None
    report(niter=10, outfile=outfile)
