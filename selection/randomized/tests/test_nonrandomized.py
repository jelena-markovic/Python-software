from __future__ import print_function
import numpy as np
from selection.tests.instance import gaussian_instance,logistic_instance
import regreg.api as rr

from selection.randomized.M_estimator_nonrandom import M_estimator

from selection.tests.flags import SET_SEED, SMALL_SAMPLES
from selection.tests.decorators import (wait_for_return_value,
                                        set_seed_iftrue,
                                        set_sampling_params_iftrue,
                                        register_report)
import selection.tests.reports as reports
from selection.randomized.query import naive_confidence_intervals, naive_pvalues


@register_report(['pivots_clt', 'covered_clt', 'ci_length_clt',
                  'pivots_naive', 'covered_naive', 'ci_length_naive'])
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
@set_seed_iftrue(SET_SEED)
@wait_for_return_value()
def test_nonrandomized(s=0,
                       n=500,
                       p=500,
                       snr=7,
                       rho=0.,
                       lam_frac=2.2,
                       loss='gaussian',
                       parametric=True,
                       ndraw = 10000,
                       burnin= 2000,
                       solve_args={'min_its': 20, 'tol': 1.e-10}):

    if loss == "gaussian":
        X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=rho, snr=snr, sigma=1)
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
        loss = rr.glm.gaussian(X, y)
    elif loss == "logistic":
        X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=rho, snr=snr)
        loss = rr.glm.logistic(X, y)
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))

    nonzero = np.where(beta)[0]
    print("lam", lam)
    W = np.ones(p) * lam
    #groups = np.arange(p)
    ngroups = 10
    groups = np.concatenate([np.arange(ngroups) for i in range(p / ngroups)])

    penalty = rr.group_lasso(groups,
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)
    M_est = M_estimator(lam, loss, penalty)
    M_est.solve()
    active  = M_est._overall
    nactive = np.sum(active)
    print("nactive", nactive)
    if nactive == 0:
        return None

    true_vec = beta[active]

    if set(nonzero).issubset(np.nonzero(active)[0]):

        score_mean = M_est.observed_score_state.copy()
        score_mean[:nactive] = 0 # M_est.initial_soln[active]
        # M_est.setup_sampler(score_mean = np.zeros(p))
        M_est.setup_sampler(score_mean=score_mean, parametric=parametric)

        score_sample = M_est.sample(ndraw=ndraw,
                                 burnin=burnin, stepsize=1./p)
        target_sample = score_sample[:, :nactive]
        target_observed = M_est.observed_score_state[:nactive]
        pivots = M_est.coefficient_pvalues(target_observed,
                                 parameter=true_vec,
                                 sample=target_sample)

        LU = M_est.confidence_intervals(target_observed,
                                    sample=target_sample,
                                    level=0.9)

        LU_naive = naive_confidence_intervals(M_est, target_observed)
        naive_pivots = naive_pvalues(M_est, target_observed, true_vec)

        def coverage(LU):
            L, U = LU[:, 0], LU[:, 1]
            covered = np.zeros(nactive)
            ci_length = np.zeros(nactive)

            for j in range(nactive):
                if (L[j] <= true_vec[j]) and (U[j] >= true_vec[j]):
                    covered[j] = 1
                ci_length[j] = U[j] - L[j]
            return covered, ci_length

        sel_covered, sel_length = coverage(LU)
        naive_covered, naive_length = coverage(LU_naive)

        return pivots, sel_covered, sel_length, naive_pivots, naive_covered, naive_length


def report(niter=50, **kwargs):

    condition_report = reports.reports['test_nonrandomized']
    runs = reports.collect_multiple_runs(condition_report['test'],
                                         condition_report['columns'],
                                         niter,
                                         reports.summarize_all,
                                         **kwargs)

    fig = reports.pivot_plot_plus_naive(runs)
    fig.suptitle("Nonrandomized pivots", fontsize=20)
    fig.savefig('nonrandomized_pivots.pdf')


if __name__ == '__main__':
    report()
