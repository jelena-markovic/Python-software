from __future__ import print_function
import numpy as np
from selection.tests.instance import gaussian_instance,logistic_instance
import regreg.api as rr

from selection.randomized.M_estimator import restricted_Mest
from selection.randomized.M_estimator_nonrandom import M_estimator

from selection.tests.flags import SET_SEED, SMALL_SAMPLES
from selection.tests.decorators import (wait_for_return_value,
                                        set_seed_iftrue,
                                        set_sampling_params_iftrue,
                                        register_report)
import selection.tests.reports as reports


@register_report(['covered_clt', 'ci_length_clt'])
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
@set_seed_iftrue(SET_SEED)
@wait_for_return_value()
def test_nonrandomized(s=0,
                       n=200,
                       p=50,
                       snr=7,
                       rho=0,
                       lam_frac=0.8,
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
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    M_est = M_estimator(lam, loss, penalty)
    M_est.solve()
    active  = M_est._overall
    nactive = np.sum(active)
    print("nactive", nactive)
    if nactive == 0:
        return None

    score_mean = M_est.observed_score_state.copy()
    score_mean[:nactive] = 0 #M_est.initial_soln[active]
    # M_est.setup_sampler(score_mean = np.zeros(p))
    M_est.setup_sampler(score_mean=score_mean, parametric=parametric)
    #M_est.sample(ndraw = 1000, burnin=1000, stepsize=1./p)

    #test_stat = lambda x: np.linalg.norm(x[:nactive])
    #M_est.hypothesis_test(test_stat, test_stat(M_est.observed_score_state), stepsize=1. / p)
    score_sample = M_est.sample(ndraw=ndraw,
                                 burnin=burnin, stepsize=1./p)
    target_sample = score_sample[:, :nactive]
    LU = M_est.confidence_intervals(M_est.observed_score_state[:nactive],
                                    sample=target_sample,
                                    level=0.9)

    true_vec = beta[active]
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

    return sel_covered, sel_length


def report(niter=50, **kwargs):

    condition_report = reports.reports['test_nonrandomized']
    runs = reports.collect_multiple_runs(condition_report['test'],
                                         condition_report['columns'],
                                         niter,
                                         reports.summarize_all,
                                         **kwargs)

    #fig = reports.pivot_plot_simple(runs)
    #fig.savefig('marginalized_subgrad_pivots.pdf')


if __name__ == '__main__':
    report()



   # pvals = []
   # for i in range(50):
   #     print(i)
   #     pval = test_nonrandomized()
   #     print(pval)
   #     if pval is not None:
   #         pvals.append(pval)
   #import matplotlib.pyplot as plt
   # import statsmodels.api as sm
   # fig = plt.figure()
   # ax = fig.gca()
   #ecdf = sm.distributions.ECDF(pvals)
   # G = np.linspace(0, 1)
   # F = ecdf(G)
   # ax.plot(G, F, '-o', c='b', lw=2)
   # ax.plot([0, 1], [0, 1], 'k-', lw=2)
   # ax.set_xlim([0, 1])
   # ax.set_ylim([0, 1])
   # plt.show()
