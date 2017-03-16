from __future__ import print_function
import numpy as np

import regreg.api as rr

import selection.tests.reports as reports


from selection.tests.flags import SMALL_SAMPLES, SET_SEED
from selection.api import (randomization,
                           split_glm_group_lasso,
                           split_glm_group_lasso_parametric,
                           multiple_queries,
                           glm_target)
from selection.tests.instance import (gaussian_instance, logistic_instance)
from selection.tests.decorators import wait_for_return_value, register_report, set_sampling_params_iftrue, set_seed_iftrue
from selection.randomized.glm import (normal_interval, split_interval)

from selection.distributions.intervals import intervals_from_sample


@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
@set_seed_iftrue(SET_SEED)
@wait_for_return_value()
def test_split_prediction(s=3,
                       n=200,
                       p=100,
                       snr = 10,
                       rho = 0.6,
                       split_frac = 0.7,
                       lam_frac = 4.,
                       ndraw=10000, burnin=2000,
                       intervals = 'old',
                       randomizer = 'gaussian',
                       randomizer_scale = 1.,
                       loss = 'gaussian',
                       solve_args={'min_its':50, 'tol':1.e-10},
                       check_screen =True,
                       parametric=True,
                       level = 0.90):

    if randomizer == 'laplace':
        randomizer = randomization.laplace((p,), scale=randomizer_scale)
    elif randomizer == 'gaussian':
        randomizer = randomization.isotropic_gaussian((p,), randomizer_scale)
    elif randomizer == 'logistic':
        randomizer = randomization.logistic((p,), scale=randomizer_scale)

    if loss == "gaussian":
        X_all, y_all, beta, nonzero, sigma = gaussian_instance(n=n + 1, p=p, s=s, rho=rho, snr=snr, sigma=1,
                                                               equi_correlated=False, random_signs=False)
        X, y = X_all[:n, :], y_all[:n]
        X_new, y_new = X_all[n, :], y_all[n]
        lam = np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 1000))))) * sigma
        loss = rr.glm.gaussian(X, y)
        loss_rr = rr.glm.gaussian
    elif loss == "logistic":
        X_all, y_all, beta, _ = logistic_instance(n=n + 1, p=p, s=s, rho=rho, snr=snr, random_signs=False)
        X, y = X_all[:n, :], y_all[:n]
        X_new, y_new = X_all[n, :], y_all[n]
        loss = rr.glm.logistic(X, y)
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))
        loss_rr = rr.glm.logistic

    nonzero = np.where(beta)[0]
    epsilon = 1./np.sqrt(n)

    W = np.ones(p)*lam
    # W[0] = 0 # use at least some unpenalized
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    m = int(split_frac * n)

    if parametric==True:
        M_est1 = split_glm_group_lasso_parametric(loss, epsilon, m, penalty)
    else:
        M_est1 = split_glm_group_lasso(loss, epsilon, m, penalty)

    mv = multiple_queries([M_est1])
    mv.solve()

    active_union = M_est1.selection_variable['variables'] #+ M_est2.selection_variable['variables']
    nactive = np.sum(active_union)
    print("nactive", nactive)
    if nactive==0:
        return None

    leftout_indices = M_est1.randomized_loss.saturated_loss.case_weights == 0

    screen = set(nonzero).issubset(np.nonzero(active_union)[0])

    if check_screen and (not screen):
        return None

    if True:
        active_set = np.nonzero(active_union)[0]
        true_vec = beta[active_union]
        linear_func = X_new[active_union]
        true_prediction_mean = np.dot(linear_func, true_vec)

        target_sampler, target_observed = glm_target(loss,
                                                     active_union,
                                                     queries=mv,
                                                     bootstrap=False,
                                                     parametric=parametric)

        if intervals == 'old':
            target_sample = target_sampler.sample(ndraw=ndraw,
                                                    burnin=burnin)
            intervals_instance = intervals_from_sample(reference=target_observed,
                                                       sample=target_sample,
                                                       observed=target_observed,
                                                       covariance=target_sampler.target_cov)

            LU_prediction_sel = intervals_instance.confidence_interval(linear_func=linear_func, level=level)
            if LU_prediction_sel is None:
                return None
        else:
            full_sample = target_sampler.sample(ndraw=ndraw,
                                                burnin=burnin,
                                                keep_opt=True)
            LU = target_sampler.confidence_intervals_translate(target_observed,
                                                               sample=full_sample)
            pivots = target_sampler.coefficient_pvalues_translate(target_observed,
                                                                        parameter=true_vec,
                                                                        sample=full_sample)

            raise Exception('new intervals not implemented for prediction')

        # naive prediction interval
        LU_prediction_naive = normal_interval(target_observed[:nactive],
                                              target_sampler.target_cov[:nactive,:nactive],
                                              linear_func=linear_func, alpha=1-level)

        if X.shape[0] - leftout_indices.sum() > nactive:
            sub_loss = loss_rr(X[leftout_indices,:], y[leftout_indices])
            LU_prediction_split = split_interval(sub_loss, active_union, linear_func, 1-level, parametric)
        else:
            LU_prediction_split = None
            return None

        def coverage(LU):
            L, U = LU[0], LU[1]
            covered = 0
            if (L <= true_prediction_mean) and (U >= true_prediction_mean):
                covered = 1
            ci_length = U - L
            return covered, ci_length

        covered_sel, ci_length_sel = coverage(LU_prediction_sel)
        covered_naive, ci_length_naive = coverage(LU_prediction_naive)
        covered_split, ci_length_split = coverage(LU_prediction_split)

        active_var = np.zeros(nactive, np.bool)
        for j in range(nactive):
            active_var[j] = active_set[j] in nonzero

        return (covered_sel, ci_length_sel), (covered_naive, ci_length_naive), \
               (covered_split, ci_length_split)


if __name__=='__main__':
    niters = 100
    selective_sample = []
    split_sample = []
    naive_sample = []
    for i in range(niters):
        print("iteration", i)
        result = test_split_prediction()[1]
        if result is not None:
            sel, naive, split = result
            selective_sample.append(sel)
            split_sample.append(split)
            naive_sample.append(naive)

            print("selective prediction ci coverage", np.mean([selective_sample[i][0] for i in range(len(selective_sample))]))
            print("selection prediction ci length", np.mean([selective_sample[i][1] for i in range(len(selective_sample))]))
            print("split prediction ci coverage", np.mean([split_sample[i][0] for i in range(len(split_sample))]))
            print("split prediction ci length", np.mean([split_sample[i][1] for i in range(len(split_sample))]))
            print("naive prediction ci coverage", np.mean([naive_sample[i][0] for i in range(len(naive_sample))]))
            print("naive prediction ci length", np.mean([naive_sample[i][1] for i in range(len(naive_sample))]))