from __future__ import print_function
import numpy as np

import regreg.api as rr
import selection.tests.reports as reports
from selection.distributions.intervals import intervals_from_sample

from selection.tests.flags import SET_SEED, SMALL_SAMPLES
from selection.tests.instance import logistic_instance, gaussian_instance
from selection.tests.decorators import (wait_for_return_value,
                                        set_seed_iftrue,
                                        set_sampling_params_iftrue,
                                        register_report)
import selection.tests.reports as reports

from selection.api import (randomization,
                           glm_group_lasso,
                           multiple_queries,
                           glm_group_lasso_parametric,
                           glm_target)

from selection.randomized.glm import normal_interval

@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
@set_seed_iftrue(SET_SEED)
@wait_for_return_value()
def test_marginalize(s=3,
                    n=100,
                    p=200,
                    rho=0.6,
                    snr=10,
                    lam_frac = 4.,
                    ndraw=10000,
                    burnin=2000,
                    loss='gaussian',
                    randomizer = 'gaussian',
                    randomizer_scale = 1.,
                    nviews=1,
                    scalings=False,
                    subgrad =True,
                    parametric=True,
                    intervals='old',
                    level=0.90,
                    linear_func=None):

    print(n,p,s)

    if randomizer == 'laplace':
        randomizer = randomization.laplace((p,), scale=randomizer_scale)
    elif randomizer == 'gaussian':
        randomizer = randomization.isotropic_gaussian((p,), randomizer_scale)
    elif randomizer == 'logistic':
        randomizer = randomization.logistic((p,), scale=randomizer_scale)

    if loss=="gaussian":
        X_all, y_all, beta, nonzero, sigma = gaussian_instance(n=n+1, p=p, s=s, rho=rho, snr=snr, sigma=1,
                                                               equi_correlated=False, random_signs=False)
        X, y = X_all[:n,:], y_all[:n]
        X_new, y_new = X_all[n,:], y_all[n]
        lam = np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 1000))))) * sigma
        loss = rr.glm.gaussian(X, y)
    elif loss=="logistic":
        X_all, y_all, beta, _ = logistic_instance(n=n+1, p=p, s=s, rho=rho, snr=snr, random_signs=False)
        X, y = X_all[:n,:], y_all[:n]
        X_new, y_new = X_all[n,:], y_all[n]
        loss = rr.glm.logistic(X, y)
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))

    epsilon = 1. / np.sqrt(n)
    W = lam_frac*np.ones(p)*lam
    #W[0] = 0 # use at least some unpenalized
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    views = []
    for i in range(nviews):
        if parametric==False:
            views.append(glm_group_lasso(loss, epsilon, penalty, randomizer))
        else:
            views.append(glm_group_lasso_parametric(loss, epsilon, penalty, randomizer))

    queries = multiple_queries(views)
    queries.solve()

    active_union = np.zeros(p, np.bool)
    for view in views:
        active_union += view.selection_variable['variables']

    nactive = np.sum(active_union)
    print("nactive", nactive)

    nonzero = np.where(beta)[0]
    true_vec = beta[active_union]

    if set(nonzero).issubset(np.nonzero(active_union)[0]):
        check_screen=True

        if nactive==s:
            return None

        if scalings: # try condition on some scalings
            for i in range(nviews):
                views[i].condition_on_subgradient()
                views[i].condition_on_scalings()
        if subgrad:
            for i in range(nviews):
               conditioning_groups = np.zeros(p,dtype=bool)
               conditioning_groups[:(p/2)] = True
               marginalizing_groups = np.zeros(p, dtype=bool)
               marginalizing_groups[(p/2):] = True
               views[i].decompose_subgradient(conditioning_groups=np.zeros(p, dtype=bool), marginalizing_groups=np.ones(p, bool))

        active_set = np.nonzero(active_union)[0]
        linear_func = X_new[active_union]

        target_sampler, target_observed = glm_target(loss,
                                                     active_union,
                                                     queries,
                                                     bootstrap=False,
                                                     parametric=parametric)
                                                     #reference= beta[active_union])

        if intervals=='old':
            target_sample = target_sampler.sample(ndraw=ndraw,
                                                  burnin=burnin)
            #LU = target_sampler.confidence_intervals(target_observed,
            #                                         sample=target_sample,
            #                                         level=0.9)
            #pivots = target_sampler.coefficient_pvalues(target_observed,
            #                                            parameter=true_vec,
            #                                            sample=target_sample)

            intervals_instance = intervals_from_sample(reference=target_observed,
                                                       sample=target_sample,
                                                       observed=target_observed,
                                                       covariance=target_sampler.target_cov)

            LU_prediction_sel = intervals_instance.confidence_interval(linear_func=linear_func, level=level)

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
            raise Exception('new intervals not implemented for prediction')

        true_prediction_mean = np.dot(linear_func, true_vec)
        def coverage(LU):
            L, U = LU[0], LU[1]
            covered = 0
            if (L <= true_prediction_mean) and (U >= true_prediction_mean):
                covered = 1
            ci_length = U - L
            return covered, ci_length

        LU_prediction_naive = normal_interval(target_observed[:nactive],
                                              target_sampler.target_cov[:nactive,:nactive],
                                              linear_func, 1-level)
        covered_sel, ci_length_sel = coverage(LU_prediction_sel)
        covered_naive, ci_length_naive = coverage(LU_prediction_naive)


        return (covered_sel, ci_length_sel), (covered_naive, ci_length_naive)


if __name__ == '__main__':
    sel_sample = []
    naive_sample = []
    length_sample = []
    niters = 100

    for i in range(niters):
        print("iteration", i)
        result = test_marginalize()[1]
        if result is not None:
            sel_sample.append(result[0])
            naive_sample.append(result[1])
            print("selective coverage", np.mean([sel_sample[i][0] for i in range(len(sel_sample))]))
            print("selective length", np.mean([sel_sample[i][1] for i in range(len(sel_sample))]))
            print("naive coverage", np.mean([naive_sample[i][0] for i in range(len(naive_sample))]))
            print("naive length", np.mean([naive_sample[i][1] for i in range(len(naive_sample))]))