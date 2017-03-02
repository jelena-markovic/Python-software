import functools # for bootstrap partial mapping
import numpy as np
import regreg.api as rr
from selection.randomized.M_estimator import restricted_Mest
from selection.randomized.glm import bootstrap_cov
from scipy.stats import norm as ndist



class LOCO(object):
    def __init__(self,
                 loss, loss_1, loss_2,
                 loss_rr,
                 active,
                 epsilon,
                 lam,
                 solve_args={'min_its': 50, 'tol': 1.e-10}):
        """
        pairs bootstrap for the loco parameter
        """
        (self.loss_rr, self. active, self.epsilon, self.solve_args) = (loss_rr, active, epsilon, solve_args)
        self.X, self.y = loss.data
        self.X1, self.y1 = loss_1.data
        self.X2, self.y2 = loss_2.data
        n1, self.p = self.X1.shape
        self.n, _ = self.X.shape
        self.nactive = active.sum()
        self.active_set =  np.nonzero(active)[0]


        def active_train(loss, lam=lam):
            X, _ = loss.data
            ncol = X.shape[1]
            penalty = rr.l1norm(ncol, lagrange=lam)
            problem = rr.simple_problem(loss, penalty)
            beta_train = problem.solve(**solve_args)
            active = beta_train !=0
            beta_bar = restricted_Mest(loss, active, solve_args=self.solve_args)
            return active, beta_bar

            #active_ = _beta(glm_loss(X1,y1))
        locos_active = []
        locos_estimators = []
        for j in range(self.nactive):
            keep = np.ones(self.p, np.bool)
            keep[self.active_set[j]] = 0
            #active, beta_bar = active_train(self.loss_rr(self.X1[:, keep], self.y1))
            active = self.active.copy()
            active = active[keep]
            beta_bar = restricted_Mest(self.loss_rr(self.X1[:, keep], self.y1), active, solve_args=self.solve_args)
            locos_active.append(active)
            locos_estimators.append(beta_bar)


        self.active_locos = locos_active
        self.locos_estimators = locos_estimators

        self.estimator = restricted_Mest(loss_1, self.active, solve_args=self.solve_args)

    def _boot_loco(self, X, y, indices):
        X_star = X[indices,:]
        y_star = y[indices]

        beta_bar = restricted_Mest(self.loss_rr(X_star, y_star), self.active, solve_args=self.solve_args)

        loco = np.zeros(self.nactive)
        for j in range(self.nactive):
            keep = np.ones(self.p, np.bool)
            keep[self.active_set[j]] = 0

            beta_bar_loco = restricted_Mest(self.loss_rr(X_star[:,keep], y_star),
                                            self.active_locos[j], solve_args=self.solve_args)

            def test_error(design, response):
                randomization = np.random.uniform(low=-1, high=1, size=X.shape[0]) * self.epsilon
                design_keep = design[:,keep]
                return np.sum(np.abs(response - np.dot(design_keep[:, self.active_locos[j]], beta_bar_loco)) \
                       - np.abs(response - np.dot(design[:, self.active], beta_bar)) + randomization)/self.n

            loco[j] = test_error(X_star, y_star)
        return loco


    def pairs_bootstrap_loco(self):
        observed = self._boot_loco(self.X, self.y, np.arange(self.n))
        print(observed)
        return functools.partial(self._boot_loco, self.X, self.y), observed


    def _split_boot_loco(self, X2, y2, indices2):
        X2_star = X2[indices2, :]
        y2_star = y2[indices2]
        n2 = X2.shape[0]

        loco = np.zeros(self.nactive)
        for j in range(self.nactive):
            keep = np.ones(self.p, np.bool)
            keep[self.active_set[j]] = 0

            def test_error(design, response):
                randomization = np.random.uniform(low=-1, high=1, size=n2) * self.epsilon
                design_keep = design[:, keep]
                return np.sum(np.abs(response - np.dot(design_keep[:, self.active_locos[j]], self.locos_estimators[j])) \
                              - np.abs(response - np.dot(design[:, self.active], self.estimator)) + randomization)

            loco[j] = test_error(X2_star, y2_star)/float(n2)
            # loco[j] = np.true_divide(sum([test_error(np.array(X2_star[i,:]), y2_star[i]) for i in range(n2)]), n2)
            # loco[j] = sum([test_error(np.array(X_star[i,:]), y_star[i]) for i in range(n)])/n
        return loco


    def split_intervals(self, alpha=0.1):
        observed = self._split_boot_loco(self.X2, self.y2, np.arange(self.X2.shape[0]))
        n2 = self.X2.shape[0]
        sampler = lambda: np.random.choice(n2, size=(n2,), replace=True)
        boot_target = functools.partial(self._split_boot_loco, self.X2, self.y2)
        cov = bootstrap_cov(sampler=sampler, boot_target=boot_target, nsample = 500)

        LU = np.zeros((2, observed.shape[0]))
        quantile = - ndist.ppf(alpha / float(2))
        for j in range(observed.shape[0]):
            sigma = np.sqrt(cov[j, j])
            LU[0, j] = observed[j] - sigma * quantile
            LU[1, j] = observed[j] + sigma * quantile
        return LU.T


def loco_target(loss,
                pairs_bootstrap_loco,
                queries,
                bootstrap = False,
                solve_args = {'min_its': 50, 'tol': 1.e-10},
                reference = None,
                parametric = False):

    """
    Form target from self.loss
    restricting to active variables.

    If subset is not None, then target returns
    only those coordinates of the active
    variables.

    Parameters
    ----------

    query : `query`
       A query with a glm loss.

    active : np.bool
       Indicators of active variables.

    queries : `multiple_queries`
       Sampler returned for this queries.

    subset : np.bool
       Indicator of subset of variables
       to be returned. Includes both
       active and inactive variables.

    bootstrap : bool
       If True, sampler returned uses bootstrap
       otherwise uses a plugin CLT.

    reference : np.float (optional)
       Optional reference parameter. Defaults
       to the observed reference parameter.
       Must have shape `active.sum()`.

    solve_args : dict
       Args used to solve restricted M estimator.

    Returns
    -------

    target_sampler : `targeted_sampler`

    """

    X, y = loss.data
    n = X.shape[0]

    boot_loco, loco_observed = pairs_bootstrap_loco

    if parametric == False:
        sampler = lambda: np.random.choice(n, size=(n,), replace=True)
        form_covariances = functools.partial(bootstrap_cov, sampler)
    else:
        raise Exception('Parametric loco not implemented yet.')
        #form_covariances = glm_parametric_covariance(loss)

    queries.setup_sampler(form_covariances)
    queries.setup_opt_state()

    if reference is None:
        reference = loco_observed

    #if parametric:
    #    linear_func = np.identity(target_observed.shape[0])
    #    _target = (active, linear_func)

    if bootstrap:
        raise Exception('Bootstrap sampler not implemented yet for loco.')
        #alpha_mat = set_alpha_matrix(loss, active, inactive=inactive)
        #alpha_subset = np.ones(alpha_mat.shape[0], np.bool)
        #alpha_subset[:nactive] = active_subset
        #alpha_mat = alpha_mat[alpha_subset]
        #target_sampler = queries.setup_bootstrapped_target(_target,
        #                                               target_observed,
        #                                               alpha_mat,
        #                                               reference=reference)
    else:
        target_sampler = queries.setup_target(boot_loco,
                                              loco_observed,
                                              reference=reference,
                                              parametric=parametric)

    return target_sampler, loco_observed


if __name__ == '__main__':
    from selection.tests.instance import gaussian_instance
    n = 200
    X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=500, s=0, rho=0, snr=1, sigma=1)
    lam = np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 1000))))) * sigma
    loss = rr.glm.gaussian(X, y)

    m=int(0.8*n)
    idx = np.zeros(n, np.bool)
    idx[:m] = 1
    np.random.shuffle(idx)
    idx = idx

    loss_1 = rr.glm.gaussian(X[idx,:], y[idx])
    loss_2 = rr.glm.gaussian(X[~idx,:], y[~idx])
    loss_rr = rr.glm.gaussian
    X, _ = loss.data
    ncol = X.shape[1]
    penalty = rr.l1norm(ncol, lagrange=lam)
    problem = rr.simple_problem(loss_1,penalty)
    beta_train = problem.solve()
    active = beta_train != 0

    _loco = LOCO(loss, loss_1, loss_2,
                loss_rr,
                active,
                epsilon=0.1,
                lam=lam)
    _loco.split_intervals()
    print(0)