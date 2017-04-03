import numpy as np
import regreg.api as rr

from selection.randomized.glm import pairs_bootstrap_glm, bootstrap_cov, _parametric_cov_glm
from ..sampling.langevin import projected_langevin
from ..distributions.api import discrete_family, intervals_from_sample

class M_estimator(object):

    def __init__(self, lam, loss, penalty, solve_args={'min_its':50, 'tol':1.e-10}):
        """
        Fits the logistic regression to a candidate active set, without penalty.
        Calls the method bootstrap_covariance() to bootstrap the covariance matrix.

        Computes $\bar{\beta}_E$ which is the restricted
        M-estimator (i.e. subject to the constraint $\beta_{-E}=0$).

        Parameters:
        -----------

        active: np.bool
            The active set from fitting the logistic lasso

        solve_args: dict
            Arguments to be passed to regreg solver.

        Returns:
        --------

        None

        Notes:
        ------

        Sets self._beta_unpenalized which will be used in the covariance matrix calculation.
        Also computes Hessian of loss at restricted M-estimator as well as the bootstrap covariance.

        """

        (self.loss,
         self.penalty,
         self.solve_args) = (loss,
                             penalty,
                             solve_args)
        self.lam = lam
    # Methods needed for subclassing a query

    def solve(self, scaling=1, solve_args={'min_its':20, 'tol':1.e-10}):

        (loss,
         penalty,
         solve_args) = (self.loss,
                        self.penalty,
                        self.solve_args)

        # initial solution

        problem = rr.simple_problem(loss, penalty)
        self.initial_soln = problem.solve(**solve_args)

        # find the active groups and their direction vectors
        # as well as unpenalized groups

        groups = np.unique(penalty.groups)
        active_groups = np.zeros(len(groups), np.bool)
        unpenalized_groups = np.zeros(len(groups), np.bool)

        active_directions = []
        active = np.zeros(loss.shape, np.bool)
        unpenalized = np.zeros(loss.shape, np.bool)

        initial_scalings = []

        self.active_directions_list = [] ## added for group lasso
        self.active_penalty = []  ## added for group lasso

        for i, g in enumerate(groups):
            group = penalty.groups == g
            active_groups[i] = (np.linalg.norm(self.initial_soln[group]) > 1.e-6 * penalty.weights[g]) and (penalty.weights[g] > 0)
            unpenalized_groups[i] = (penalty.weights[g] == 0)
            if active_groups[i]:
                active[group] = True
                z = np.zeros(active.shape, np.float)
                z[group] = self.initial_soln[group] / np.linalg.norm(self.initial_soln[group])
                active_directions.append(z)
                initial_scalings.append(np.linalg.norm(self.initial_soln[group]))
                self.active_directions_list.append(z[group])
                self.active_penalty.append(penalty.weights[g])
            if unpenalized_groups[i]:
                unpenalized[group] = True

        # solve the restricted problem

        self._overall = active + unpenalized
        self._inactive = ~self._overall
        self._unpenalized = unpenalized
        self._active_directions = np.array(active_directions).T
        self._active_groups = np.array(active_groups, np.bool)
        self._unpenalized_groups = np.array(unpenalized_groups, np.bool)

        self.selection_variable = {'groups':self._active_groups,
                                   'variables':self._overall,
                                   'directions':self._active_directions}

        # initial state for opt variables

        initial_subgrad = -self.loss.smooth_objective(self.initial_soln, 'grad')
                          # the quadratic of a smooth_atom is not included in computing the smooth_objective

        #print("initial sub", initial_subgrad)
        X, y = loss.data
        #print(np.dot(X.T, y-X.dot(self.initial_soln)))

        initial_subgrad = initial_subgrad[self._inactive]
        initial_unpenalized = self.initial_soln[self._unpenalized]

        self.observed_opt_state = np.concatenate([initial_scalings,
                                                  initial_unpenalized,
                                                  initial_subgrad], axis=0)

        # set the _solved bit

        self._solved = True

        # Now setup the pieces for linear decomposition

        (loss,
         penalty,
         initial_soln,
         overall,
         inactive,
         unpenalized,
         active_groups,
         active_directions) = (self.loss,
                               self.penalty,
                               self.initial_soln,
                               self._overall,
                               self._inactive,
                               self._unpenalized,
                               self._active_groups,
                               self._active_directions)

        # scaling should be chosen to be Lipschitz constant for gradient of Gaussian part

        # we are implicitly assuming that
        # loss is a pairs model

        _sqrt_scaling = np.sqrt(scaling)

        _beta_unpenalized = restricted_Mest(loss, overall, solve_args=solve_args)

        beta_full = np.zeros(overall.shape)
        beta_full[overall] = _beta_unpenalized
        _hessian = loss.hessian(beta_full)
        self._beta_full = beta_full

        # observed state for score

        self.observed_score_state = np.hstack([_beta_unpenalized * _sqrt_scaling,
                                               -loss.smooth_objective(beta_full, 'grad')[inactive] / _sqrt_scaling])
        #print('observed score', self.observed_score_state)
        #print("obs score", self.observed_score_state[])
        # print()

        #print(self.observed_score_state.shape)
        # form linear part

        self.num_opt_var = p = loss.shape[0] # shorthand for p

        # (\bar{\beta}_{E \cup U}, N_{-E}, c_E, \beta_U, z_{-E})
        # E for active
        # U for unpenalized
        # -E for inactive

        _opt_linear_term = np.zeros((p, self._active_groups.sum() + unpenalized.sum() + inactive.sum()))
        _score_linear_term = np.zeros((p, p))

        # \bar{\beta}_{E \cup U} piece -- the unpenalized M estimator

        Mest_slice = slice(0, overall.sum())
        _Mest_hessian = _hessian[:,overall]
        _score_linear_term[:,Mest_slice] = -_Mest_hessian / _sqrt_scaling

        # N_{-(E \cup U)} piece -- inactive coordinates of score of M estimator at unpenalized solution

        null_idx = range(overall.sum(), p)
        inactive_idx = np.nonzero(inactive)[0]
        for _i, _n in zip(inactive_idx, null_idx):
            _score_linear_term[_i,_n] = -_sqrt_scaling

        #print("score mat", _score_linear_term)
        # c_E piece

        scaling_slice = slice(0, active_groups.sum())
        if len(active_directions)==0:
            _opt_hessian=0
        else:
            epsilon=0
            _opt_hessian = (_hessian + epsilon * np.identity(p)).dot(active_directions)
        _opt_linear_term[:,scaling_slice] = _opt_hessian / _sqrt_scaling

        self.observed_opt_state[scaling_slice] *= _sqrt_scaling

        # beta_U piece

        unpenalized_slice = slice(active_groups.sum(), active_groups.sum() + unpenalized.sum())
        unpenalized_directions = np.identity(p)[:,unpenalized]
        if unpenalized.sum():
            epsilon = 0
            _opt_linear_term[:,unpenalized_slice] = (_hessian + epsilon * np.identity(p)).dot(unpenalized_directions) / _sqrt_scaling

        self.observed_opt_state[unpenalized_slice] *= _sqrt_scaling

        # subgrad piece

        subgrad_idx = range(active_groups.sum() + unpenalized.sum(), active_groups.sum() + inactive.sum() + unpenalized.sum())
        subgrad_slice = slice(active_groups.sum() + unpenalized.sum(), active_groups.sum() + inactive.sum() + unpenalized.sum())
        for _i, _s in zip(inactive_idx, subgrad_idx):
            _opt_linear_term[_i,_s] = _sqrt_scaling

        self.observed_opt_state[subgrad_slice] /= _sqrt_scaling

        # form affine part

        _opt_affine_term = np.zeros(p)
        idx = 0
        groups = np.unique(penalty.groups)
        for i, g in enumerate(groups):
            if active_groups[i]:
                group = penalty.groups == g
                _opt_affine_term[group] = active_directions[:,idx][group] * penalty.weights[g]
                idx += 1

        # two transforms that encode score and optimization
        # variable roles

        # later, we will modify `score_transform`
        # in `linear_decomposition`

        self.opt_transform = (_opt_linear_term, _opt_affine_term)

        self.score_transform = (_score_linear_term, np.zeros(_score_linear_term.shape[0]))

        # now store everything needed for the projections
        # the projection acts only on the optimization
        # variables

        self.scaling_slice = scaling_slice

        # weights are scaled here because the linear terms scales them by scaling

        new_groups = penalty.groups[inactive]
        new_weights = dict([(g, penalty.weights[g] / _sqrt_scaling) for g in penalty.weights.keys() if g in np.unique(new_groups)])

        # we form a dual group lasso object
        # to do the projection

        self.group_lasso_dual = rr.group_lasso_dual(new_groups, weights=new_weights, bound=1.)
        self.subgrad_slice = subgrad_slice

        self._setup = True
        self.Q = (_hessian[:, active])[active, :]
        self.Qinv = np.linalg.inv(self.Q)
        self.form_VQLambda()


    def projection(self, opt_state):
        """
        Full projection for Langevin.

        The state here will be only the state of the optimization variables.
        """

        if not self._setup:
            raise ValueError('setup_sampler should be called before using this function')

        new_state = opt_state.copy() # not really necessary to copy
        new_state[self.scaling_slice] = np.maximum(opt_state[self.scaling_slice], 0)
        new_state[self.subgrad_slice] = self.group_lasso_dual.bound_prox(opt_state[self.subgrad_slice])
        return new_state


    def normal_data_gradient(self, data_vector):
        return -np.dot(self.total_cov_inv, data_vector-self.reference)

    def gradient(self, opt_state):
        """
        Randomization derivative at full state.
        """

        opt_linear, opt_offset = self.opt_transform
        opt_piece = opt_linear.dot(opt_state) + opt_offset

        # value of the data D
        #full_state = self.score_transform_inv.dot(opt_piece)

        data_derivative = self.normal_data_gradient(opt_piece)
        # chain rule for optimization part

        opt_grad = opt_linear.T.dot(data_derivative)
        opt_grad += self.derivative_logdet(opt_state[self.scaling_slice])

        return opt_grad

    def setup_sampler(self, score_mean,
                      parametric = False,
                      scaling=1, solve_args={'min_its':20, 'tol':1.e-10}):

        X, _ = self.loss.data
        n = X.shape[0]
        if (parametric==False):
            bootstrap_score = pairs_bootstrap_glm(self.loss,
                                              self._overall,
                                              beta_full=self._beta_full,
                                              inactive=~self._overall)[0]
            score_cov = bootstrap_cov(lambda: np.random.choice(n, size=(n,), replace=True), bootstrap_score)
        else:
            score_cov = _parametric_cov_glm(self.loss,
                                            self._overall,
                                            beta_full=self._beta_full,
                                            inactive=~self._overall)

        self.score_cov = score_cov
        self.score_cov_inv = np.linalg.inv(self.score_cov)

        self.score_mat = self.score_transform[0]
        self.score_mat_inv = np.linalg.inv(-self.score_transform[0])
        self.total_cov = np.dot(self.score_mat, self.score_cov).dot(self.score_mat.T)
        self.total_cov_inv = np.linalg.inv(self.total_cov)
        self.reference = self.score_mat.dot(score_mean)

        nactive = self._overall.sum()
        self.target_cov = self.score_cov[:nactive,:nactive]
        self.shape = (nactive,1)

    def form_VQLambda(self):
        nactive_groups = len(self.active_directions_list)
        nactive_vars = np.sum([self.active_directions_list[i].shape[0] for i in range(nactive_groups)])
        V = np.zeros((nactive_vars, nactive_vars - nactive_groups))
        # U = np.zeros((nvariables, ngroups))
        Lambda = np.zeros((nactive_vars, nactive_vars))
        temp_row, temp_col = 0, 0
        for g in range(len(self.active_directions_list)):
            size_curr_group = self.active_directions_list[g].shape[0]
            # U[temp_row:(temp_row+size_curr_group),g] = self._active_directions[g]
            Lambda[temp_row:(temp_row + size_curr_group), temp_row:(temp_row + size_curr_group)] \
                = self.active_penalty[g] * np.identity(size_curr_group)
            import scipy
            from scipy import linalg, matrix
            def null(A, eps=1e-12):
                u, s, vh = scipy.linalg.svd(A)
                padding = max(0, np.shape(A)[1] - np.shape(s)[0])
                null_mask = np.concatenate(((s <= eps), np.ones((padding,), dtype=bool)), axis=0)
                null_space = scipy.compress(null_mask, vh, axis=0)
                return scipy.transpose(null_space)

            V_g = null(matrix(self.active_directions_list[g]))
            V[temp_row:(temp_row + V_g.shape[0]), temp_col:(temp_col + V_g.shape[1])] = V_g
            temp_row += V_g.shape[0]
            temp_col += V_g.shape[1]
        self.VQLambda = np.dot(np.dot(V.T, self.Qinv), Lambda.dot(V))


    def derivative_logdet(self, scalings):
        nactive_groups = len(self.active_directions_list)
        nactive_vars = np.sum([self.active_directions_list[i].shape[0] for i in range(nactive_groups)])
        from scipy.linalg import block_diag
        matrix_list = [scalings[i] * np.identity(self.active_directions_list[i].shape[0] - 1) for i in
                       range(scalings.shape[0])]
        Gamma_minus = block_diag(*matrix_list)
        jacobian_inv = np.linalg.inv(Gamma_minus + self.VQLambda)

        group_sizes = [self._active_directions[i].shape[0] for i in range(nactive_groups)]
        group_sizes_cumsum = np.concatenate(([0], np.array(group_sizes).cumsum()))

        jacobian_inv_blocks = [jacobian_inv[group_sizes_cumsum[i]:group_sizes_cumsum[i + 1],
                               group_sizes_cumsum[i]:group_sizes_cumsum[i + 1]]
                               for i in range(nactive_groups)]

        der = np.zeros(self.observed_opt_state.shape[0])
        der[self.scaling_slice] = np.array([np.matrix.trace(jacobian_inv_blocks[i]) for i in range(scalings.shape[0])])
        return der


    def reconstruction_map(self, opt_state):

        if not self._setup:
            raise ValueError('setup_sampler should be called before using this function')

        # reconstruction of randoimzation omega

        opt_state = np.atleast_2d(opt_state)
        opt_linear, opt_offset = self.opt_transform
        opt_piece = opt_linear.dot(opt_state.T).flatten() + opt_offset
        return self.score_mat_inv.dot(opt_piece)

    def sample(self, ndraw, burnin, stepsize):
        '''
        Sample `target` from selective density
        using projected Langevin sampler with
        gradient map `self.gradient` and
        projection map `self.projection`.

        Parameters
        ----------

        ndraw : int
           How long a chain to return?

        burnin : int
           How many samples to discard?

        stepsize : float
           Stepsize for Langevin sampler. Defaults
           to a crude estimate based on the
           dimension of the problem.

        keep_opt : bool
           Should we return optimization variables
           as well as the target?

        Returns
        -------

        gradient : np.float
        '''
        #if stepsize is None:
        #    stepsize = 1. / self.crude_lipschitz()
        langevin = projected_langevin(self.observed_opt_state.copy(),
                                      self.gradient,
                                      self.projection,
                                      stepsize)

        samples = []
        for i in range(ndraw + burnin):
            langevin.next()
            if (i >= burnin):
                samples.append(self.reconstruction_map(langevin.state.copy()))

        return np.asarray(samples)

    def hypothesis_test(self,
                        test_stat,
                        observed_value,
                        ndraw=10000,
                        burnin=2000,
                        stepsize=None,
                        sample=None,
                        parameter=None,
                        alternative='twosided'):

        '''
        Sample `target` from selective density
        using projected Langevin sampler with
        gradient map `self.gradient` and
        projection map `self.projection`.

        Parameters
        ----------

        test_stat : callable
           Test statistic to evaluate on sample from
           selective distribution.

        observed_value : float
           Observed value of test statistic.
           Used in p-value calculation.

        ndraw : int
           How long a chain to return?

        burnin : int
           How many samples to discard?

        stepsize : float
           Stepsize for Langevin sampler. Defaults
           to a crude estimate based on the
           dimension of the problem.

        sample : np.array (optional)
           If not None, assumed to be a sample of shape (-1,) + `self.shape`
           representing a sample of the target from parameters `self.reference`.
           Allows reuse of the same sample for construction of confidence
           intervals, hypothesis tests, etc. If not None,
           `ndraw, burnin, stepsize` are ignored.

        parameter : np.float (optional)
           If not None, defaults to `self.reference`.
           Otherwise, sample is reweighted using Gaussian tilting.

        alternative : ['greater', 'less', 'twosided']
            What alternative to use.

        Returns
        -------

        gradient : np.float

        '''

        if alternative not in ['greater', 'less', 'twosided']:
            raise ValueError("alternative should be one of ['greater', 'less', 'twosided']")

        if sample is None:
            sample = self.sample(ndraw, burnin, stepsize=stepsize)

        if parameter is None:
            parameter = self.reference

        sample_test_stat = np.squeeze(np.array([test_stat(x) for x in sample]))
        family = discrete_family(sample_test_stat, np.ones_like(sample_test_stat))
        pval = family.cdf(0, observed_value)

        if alternative == 'greater':
            return 1 - pval
        elif alternative == 'less':
            return pval
        else:
            return 2 * min(pval, 1 - pval)

    def confidence_intervals(self,
                            observed,
                            ndraw=10000,
                            burnin=2000,
                            stepsize=None,
                            sample=None,
                            level=0.9):


        '''
        Parameters
        ----------

        observed : np.float
            A vector of parameters with shape `self.shape`,
            representing coordinates of the target.

        ndraw : int
            How long a chain to return?

        burnin : int
            How many samples to discard?

        stepsize : float
            Stepsize for Langevin sampler. Defaults
            to a crude estimate based on the
            dimension of the problem.

        sample : np.array (optional)
            If not None, assumed to be a sample of shape (-1,) + `self.shape`
            representing a sample of the target from parameters `self.reference`.
            Allows reuse of the same sample for construction of confidence
            intervals, hypothesis tests, etc.

        level : float (optional)
            Specify the
            confidence level.

        Notes
        -----

        Construct selective confidence intervals
        for each parameter of the target.

        Returns
        -------

        intervals : [(float, float)]
        List of confidence intervals.

        '''

        if sample is None:
            sample = self.sample(ndraw, burnin, stepsize=stepsize)

        nactive = observed.shape[0]
        intervals_instance = intervals_from_sample(self.reference[:self._overall.sum()],
                                           sample,
                                           observed,
                                           self.target_cov)

        return intervals_instance.confidence_intervals_all(level=level)

    def coefficient_pvalues(self,
                            observed,
                            parameter=None,
                            ndraw=10000,
                            burnin=2000,
                            stepsize=None,
                            sample=None,
                            alternative='twosided'):
        '''
        Construct selective p-values
        for each parameter of the target.
        Parameters
        ----------
        observed : np.float
            A vector of parameters with shape `self.shape`,
            representing coordinates of the target.
        parameter : np.float (optional)
            A vector of parameters with shape `self.shape`
            at which to evaluate p-values. Defaults
            to `np.zeros(self.shape)`.
        ndraw : int
            How long a chain to return?
        burnin : int
            How many samples to discard?
        stepsize : float
            Stepsize for Langevin sampler. Defaults
            to a crude estimate based on the
            dimension of the problem.
        sample : np.array (optional)
            If not None, assumed to be a sample of shape (-1,) + `self.shape`
            representing a sample of the target from parameters `self.reference`.
            Allows reuse of the same sample for construction of confidence
            intervals, hypothesis tests, etc.
        alternative : ['greater', 'less', 'twosided']
            What alternative to use.
        Returns
        -------
        pvalues : np.float
        '''

        if alternative not in ['greater', 'less', 'twosided']:
            raise ValueError("alternative should be one of ['greater', 'less', 'twosided']")

        if sample is None:
            sample = self.sample(ndraw, burnin, stepsize=stepsize)

        if parameter is None:
            parameter = np.zeros(observed)

        nactive = observed.shape[0]
        intervals_instance = intervals_from_sample(self.reference[:self._overall.sum()],
                                           sample,
                                           observed,
                                           self.target_cov)

        pval = intervals_instance.pivots_all(parameter)

        if alternative == 'greater':
            return 1 - pval
        elif alternative == 'less':
            return pval
        else:
            return 2 * np.minimum(pval, 1 - pval)


def restricted_Mest(Mest_loss, active, solve_args={'min_its':50, 'tol':1.e-10}):

    X, Y = Mest_loss.data

    if Mest_loss._is_transform:
        raise NotImplementedError('to fit restricted model, X must be an ndarray or scipy.sparse; general transforms not implemented')
    X_restricted = X[:,active]
    loss_restricted = rr.affine_smooth(Mest_loss.saturated_loss, X_restricted)
    beta_E = loss_restricted.solve(**solve_args)

    return beta_E