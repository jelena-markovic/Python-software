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

from selection.randomized.glm import glm_parametric_covariance, glm_nonparametric_bootstrap, restricted_Mest, set_alpha_matrix


from rpy2.robjects.packages import importr
from rpy2 import robjects
# the R packages should be installed in "/Users/Jelena/anaconda/lib/R/library"
# base = importr('base')
# print(base.R_home())
# should match with .libPaths() from R
gamsel = importr('gamsel')
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

robjects.r('''
 setup_gamsel = function(x, degree, df){
  bases = pseudo.bases(x=x, degree = degree, df = df, parallel=FALSE)
  degrees = sapply(bases, dim)[2, ]
  U = do.call("cbind", bases)
  X = do.call("cbind", lapply(bases, function(x) x[, 1, drop = FALSE])) # unit norm version of x

  parms = lapply(bases, "attr", "parms")
  getdpsi = function(parms) {
    d = parms$d
    if (length(d) > 1) {
      psi = d[2]
      d = d/d[2]
      d[1] = 1
    }
    else {
      d = 1
      psi = 0
    }
    list(d = d, psi = psi)
  }
  dpsi = lapply(parms, getdpsi)
  D_seq = unlist(lapply(dpsi, "[[", "d"))
  psi = sapply(dpsi, "[[", "psi")
  return(list(D_seq = D_seq, psi=psi, U=U, degrees=degrees))
}
''')


def setup_gamsel(s, n, p, rho, signal, lam_frac,
                 degree, df, gamma):

    X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=rho, signal=signal, sigma=1)

    ## forming U, psi and D should come here
    r_setup_gamsel = robjects.globalenv['setup_gamsel']
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    result = r_setup_gamsel(r_X, degree, df)
    D_star_seq = np.array(result[0])
    psi_seq = np.array(result[1])
    U = np.array(result[2])
    degrees = np.array(result[3])

    # remove X from U:
    keep_indices = np.ones(U.shape[1], np.bool)
    ind = 0
    for i in range(p):
        keep_indices[ind] = False
        ind = ind + degrees[i]
    U = U[:, keep_indices]

    V = np.dot(U, np.diag(np.true_divide(1., np.sqrt(D_star_seq[keep_indices]))))

    intercept = np.ones((n, 1)) / np.sqrt(n)
    X_joint = np.concatenate((intercept, X, V), axis=1)
    p_joint = X_joint.shape[1]
    print("design dim", p_joint)
    beta_joint = np.concatenate((beta, np.zeros(p_joint - beta.shape[0])))

    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 1000))))) * sigma
    loss = rr.glm.gaussian(X_joint, y)

    epsilon = 1. / np.sqrt(n)
    psi_seq_extended = []
    for i in range(p):
        # add_group_penalty = np.concatenate(([0], np.ones(degrees[i]-1)*psi_seq[i]))
        add_group_penalty = np.ones(degrees[i] - 1) * psi_seq[i]
        psi_seq_extended = np.concatenate((psi_seq_extended, add_group_penalty))
    epsilon_seq = epsilon * np.ones(p_joint) + np.concatenate(
        (np.zeros(p + 1), psi_seq_extended))  # ridge_penalty_weights

    groups = np.arange(p + 1)  # initially intercept and individual predictors
    weights = dict(zip(groups, np.ones(p + 1) * lam * gamma))
    weights.update({0: 0})  # no penalty on the intercept
    for i in range(p + 1, 2 * p + 1):  # add p more groups
        group = np.ones(degrees[i - (p + 1)] - 1) * i
        groups = np.concatenate((groups, group))
        weights.update({i: lam * (1 - gamma)})
    groups = np.array(groups, int)

    print("groups", set(groups))
    print("weights", weights)
    penalty = rr.group_lasso(groups, weights=weights, lagrange=1.)
    return loss, epsilon_seq, penalty, beta, beta_joint


def coverage(LU, check_screen, true_vec):
    L, U = LU[:, 0], LU[:, 1]
    nactive = true_vec.shape[0]
    covered = np.zeros(nactive)
    ci_length = np.zeros(nactive)

    for j in range(nactive):
        if check_screen:
            if (L[j] <= true_vec[j]) and (U[j] >= true_vec[j]):
                covered[j] = 1
        else:
            covered[j] = None
        ci_length[j] = U[j] - L[j]
    return covered, ci_length


@register_report(['truth', 'covered_clt', 'ci_length_clt',
                  'naive_pvalues', 'covered_naive', 'ci_length_naive'])
@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
@set_seed_iftrue(SET_SEED)
@wait_for_return_value()
def test_gamsel_ci(s=0,
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
    print("nactive variables", nactive)
    print("active groups", np.where(views[0]._active_groups)[0])


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

        active_set = np.nonzero(active_union)[0]
        target_sampler, target_observed = glm_target(loss,
                                                     active_union,
                                                     queries,
                                                     bootstrap=False,
                                                     parametric=parametric)
                                                     #reference= beta[active_union])

        if intervals=='old':
            target_sample = target_sampler.sample(ndraw=ndraw,
                                                  burnin=burnin)
            LU = target_sampler.confidence_intervals(target_observed,
                                                     sample=target_sample,
                                                     level=0.9)
            pivots = target_sampler.coefficient_pvalues(target_observed,
                                                        parameter=true_vec,
                                                        sample=target_sample)
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

        # test_stat = lambda x: np.linalg.norm(x - beta[active_union])
        # observed_test_value = test_stat(target_observed)
        # pivots = target_sampler.hypothesis_test(test_stat,
        #                                       observed_test_value,
        #                                       alternative='twosided',
        #                                       parameter = beta[active_union],
        #                                       ndraw=ndraw,
        #                                       burnin=burnin,
        #                                       stepsize=None)


        covered, ci_length = coverage(LU, check_screen, true_vec)
        LU_naive = naive_confidence_intervals(target_sampler, target_observed)
        covered_naive, ci_length_naive = coverage(LU_naive, check_screen, true_vec)
        naive_pvals = naive_pvalues(target_sampler, target_observed, true_vec)

        return pivots, covered, ci_length, naive_pvals, covered_naive, ci_length_naive


def report(niter=50, **kwargs):

    condition_report = reports.reports['test_gamsel_ci']
    runs = reports.collect_multiple_runs(condition_report['test'],
                                         condition_report['columns'],
                                         niter,
                                         reports.summarize_all,
                                         **kwargs)

    fig = reports.pivot_plot_plus_naive(runs)
    #fig = reports.pivot_plot_2in1(runs,color='b', label='marginalized subgradient')
    fig.suptitle('gamsel_ci')
    fig.savefig('gamsel_ci.pdf')


if __name__ == '__main__':
    np.random.seed(1)
    report()