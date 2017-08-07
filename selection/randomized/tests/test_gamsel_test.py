from __future__ import print_function
import numpy as np

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
                           glm_group_lasso_epsilon_seq,
                           pairs_bootstrap_glm,
                           multiple_queries,
                           discrete_family,
                           projected_langevin,
                           glm_group_lasso_parametric,
                           glm_target)

from selection.randomized.query import (naive_pvalues, naive_confidence_intervals)
from selection.randomized.tests.test_gamsel_ci import setup_gamsel
from scipy.stats import chi2


def naive_pvalue(observed, cov, indices):
    observed_subset = observed[indices]
    cov_subset = cov[:, indices][indices,:]
    cov_subset_cholesky = np.linalg.cholesky(np.linalg.inv(cov_subset))
    observed_test_value = np.linalg.norm(np.dot(cov_subset_cholesky, observed_subset))**2
    pval = chi2.cdf(observed_test_value, df=np.sum(indices))
    return 2*min(pval, 1-pval)


@register_report(['truth', 'naive_pvalues'])
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
@set_seed_iftrue(SET_SEED)
@wait_for_return_value()
def test_gamsel_test(s=0,
                    n=300,
                    p=10,
                    rho=0.,
                    signal=3.5,
                    lam_frac = 4.,
                    degree = 3,
                    df=2,
                    gamma=0.4,
                    ndraw=10000,
                    burnin=2000,
                    loss='gaussian',
                    randomizer = 'gaussian',
                    randomizer_scale = 1.,
                    nviews=1,
                    scalings=False,
                    subgrad = True,
                    parametric=False,
                    intervals='old'):

    print(n,p,s)

    if loss=="gaussian":
        loss, epsilon_seq, penalty, beta, beta_joint = setup_gamsel(s=s, n=n, p=p, rho=rho, signal=signal,
                                                             lam_frac=lam_frac, degree=degree, df=df, gamma=gamma)

    elif loss=="logistic":
        X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=rho, signal=signal)
        loss = rr.glm.logistic(X, y)
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))

    X_joint, _ = loss.data
    p_joint = X_joint.shape[1]

    if randomizer == 'laplace':
        randomizer = randomization.laplace((p_joint,), scale=randomizer_scale)
    elif randomizer == 'gaussian':
        randomizer = randomization.isotropic_gaussian((p_joint,), randomizer_scale)
    elif randomizer == 'logistic':
        randomizer = randomization.logistic((p_joint,), scale=randomizer_scale)

    views = []
    for i in range(nviews):
        if parametric==False:
            views.append(glm_group_lasso_epsilon_seq(loss, epsilon_seq, penalty, randomizer))
        else:
            views.append(glm_group_lasso_parametric(loss, epsilon_seq, penalty, randomizer))

    queries = multiple_queries(views)
    queries.solve()

    active_union = np.zeros(p_joint, np.bool)
    for view in views:
        active_union += view.selection_variable['variables']

    nactive = np.sum(active_union)
    active_set = np.nonzero(active_union)[0]
    print("nactive vars", nactive)
    print("active vars", active_set)
    print("all groups", penalty.groups)
    active_union_groups = penalty.groups[active_union]

    active_groups_set = np.unique(penalty.groups[active_union])
    print("active groups", active_groups_set)
    active_add_groups_set = [active_groups_set[i] for i in range(active_groups_set.shape[0]) if active_groups_set[i]>p]
    print("active addditive groups", active_add_groups_set)
    if len(active_add_groups_set)==0:
        return None

    interest_group_vars = (penalty.groups==active_add_groups_set[0])
    interest_group_active_vars = (active_union_groups==active_add_groups_set[0])
    print("interest group vars ", np.where(interest_group_vars)[0])
    print("interest group active var", np.where(interest_group_active_vars)[0])

    nonzero = np.where(beta)[0]
    true_vec = beta_joint[active_union]

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
               views[i].decompose_subgradient(conditioning_groups=np.zeros(p_joint, dtype=bool),
                                              marginalizing_groups=np.ones(p_joint, bool))

        reference = views[0].observed_score_state[:nactive]
        reference[interest_group_active_vars] = 0
        target_sampler, target_observed = glm_target(loss,
                                                     active_union,
                                                     queries,
                                                     bootstrap=False,
                                                     parametric=parametric,
                                                     reference=reference)

        target_sample = target_sampler.sample(ndraw=ndraw,
                                             burnin=burnin)

        test_stat = lambda x: np.linalg.norm(x[interest_group_active_vars])
        observed_test_value = test_stat(target_observed)
        sample_test_stat = np.squeeze(np.array([test_stat(x) for x in target_sample]))
        family = discrete_family(sample_test_stat, np.ones(target_sample.shape[0]))
        pval = family.cdf(0, observed_test_value)

        naive_pval = naive_pvalue(target_observed, target_sampler.target_cov, interest_group_active_vars)

        return [pval], [naive_pval]


def report(niter=100, **kwargs):

    condition_report = reports.reports['test_gamsel_test']
    runs = reports.collect_multiple_runs(condition_report['test'],
                                         condition_report['columns'],
                                         niter,
                                         reports.summarize_all,
                                         **kwargs)

    fig = reports.pivot_plot_plus_naive(runs)
    #fig = reports.pivot_plot_2in1(runs,color='b', label='marginalized subgradient')
    fig.suptitle('gamsel_test')
    fig.savefig('gamsel_test.pdf')


if __name__ == '__main__':
    np.random.seed(1)
    report()