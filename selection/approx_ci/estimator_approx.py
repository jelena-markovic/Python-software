import numpy as np
from selection.randomized.M_estimator import (M_estimator, restricted_Mest)
from selection.randomized.glm import pairs_bootstrap_glm, bootstrap_cov
from selection.randomized.threshold_score import threshold_score
from selection.randomized.greedy_step import greedy_score_step

class M_estimator_approx(M_estimator):

    def __init__(self, loss, epsilon, penalty, randomization, randomizer):
        M_estimator.__init__(self, loss, epsilon, penalty, randomization)
        self.randomizer = randomizer

    def solve_approx(self):
        self.solve()
        (_opt_linear_term, _opt_affine_term) = self.opt_transform
        self._opt_linear_term = np.concatenate(
            (_opt_linear_term[self._overall, :], _opt_linear_term[~self._overall, :]), 0)
        self._opt_affine_term = np.concatenate((_opt_affine_term[self._overall], _opt_affine_term[~self._overall]), 0)
        self.opt_transform = (self._opt_linear_term, self._opt_affine_term)

        (_score_linear_term, _) = self.score_transform
        self._score_linear_term = np.concatenate(
            (_score_linear_term[self._overall, :], _score_linear_term[~self._overall, :]), 0)
        self.score_transform = (self._score_linear_term, np.zeros(self._score_linear_term.shape[0]))
        self.feasible_point = np.abs(self.initial_soln[self._overall])
        lagrange = []
        for key, value in self.penalty.weights.iteritems():
            lagrange.append(value)
        lagrange = np.asarray(lagrange)
        self.inactive_lagrange = lagrange[~self._overall]

        X, _ = self.loss.data
        n, p = X.shape
        self.p = p
        bootstrap_score = pairs_bootstrap_glm(self.loss,
                                              self._overall,
                                              beta_full=self._beta_full,
                                              inactive=~self._overall)[0]

        score_cov = bootstrap_cov(lambda: np.random.choice(n, size=(n,), replace=True), bootstrap_score)
        nactive = self._overall.sum()
        self.score_target_cov = score_cov[:, :nactive]
        self.target_cov = score_cov[:nactive, :nactive]
        self.target_observed = self.observed_score_state[:nactive]
        self.nactive = nactive

        self.B_active = self._opt_linear_term[:nactive, :nactive]
        self.B_inactive = self._opt_linear_term[nactive:, :nactive]


    def setup_map(self, j):

        self.A = np.dot(self._score_linear_term, self.score_target_cov[:, j]) / self.target_cov[j, j]
        self.null_statistic = self._score_linear_term.dot(self.observed_score_state) - self.A * self.target_observed[j]

        self.offset_active = self._opt_affine_term[:self.nactive] + self.null_statistic[:self.nactive]
        self.offset_inactive = self.null_statistic[self.nactive:]


    def bootstrap_sample(self, solve_args={'min_its':20, 'tol':1.e-10}):
        X, y = self.loss.data
        n = X.shape[0]
        sampler = lambda: np.random.choice(n, size=(n,), replace=True)
        indices = sampler()
        _boot_loss = self.loss.subsample(indices)
        _beta_unpenalized = restricted_Mest(_boot_loss, self._overall)
        return _beta_unpenalized


    def bootstrap_sample_new(self, j):
        X, y = self.loss.data
        XE = X[:,self._overall]
        n = X.shape[0]
        sampler = lambda: np.random.choice(n, size=(n,), replace=True)
        indices = sampler()
        return np.linalg.lstsq(XE[indices, :], y[indices])[0][j]


class threshold_score_approx(threshold_score):

    def __init__(self, loss,
                 threshold,
                 randomization,
                 active_bool,
                 inactive_bool,
                 randomizer):

        threshold_score.__init__(self, loss, threshold, randomization, active_bool, inactive_bool)
        self.randomizer = randomizer

    def solve_approx(self):
        self.solve()
        self.setup_sampler()
        self.feasible_point = self.observed_opt_state[self.boundary]
        (_opt_linear_term, _opt_offset) = self.opt_transform
        self._opt_linear_term = np.concatenate((_opt_linear_term[self.boundary, :], _opt_linear_term[self.interior, :]),
                                               0)
        self._opt_affine_term = np.concatenate((_opt_offset[self.boundary], _opt_offset[self.interior]), 0)
        self.opt_transform = (self._opt_linear_term, self._opt_affine_term)

        (_score_linear_term, _) = self.score_transform
        self._score_linear_term = np.concatenate(
            (_score_linear_term[self.boundary, :], _score_linear_term[self.interior, :]), 0)
        self.score_transform = (self._score_linear_term, np.zeros(self._score_linear_term.shape[0]))
        self._overall = self.boundary
        self.inactive_lagrange = self.threshold[0] * np.ones(np.sum(~self.boundary))

        X, _ = self.loss.data
        n, p = X.shape
        self.p = p
        bootstrap_score = pairs_bootstrap_glm(self.loss,
                                              self._overall,
                                              beta_full=self._beta_full,
                                              inactive=~self._overall)[0]

        score_cov = bootstrap_cov(lambda: np.random.choice(n, size=(n,), replace=True), bootstrap_score)
        nactive = self._overall.sum()
        self.score_target_cov = score_cov[:, :nactive]
        self.target_cov = score_cov[:nactive, :nactive]
        self.target_observed = self.observed_score_state[:nactive]
        self.nactive = nactive

        self.B_active = self._opt_linear_term[:nactive, :nactive]
        self.B_inactive = self._opt_linear_term[nactive:, :nactive]


    def setup_map(self, j):

        self.A = np.dot(self._score_linear_term, self.score_target_cov[:, j]) / self.target_cov[j, j]
        self.null_statistic = self._score_linear_term.dot(self.observed_score_state) - self.A * self.target_observed[j]

        self.offset_active = self._opt_affine_term[:self.nactive] + self.null_statistic[:self.nactive]
        self.offset_inactive = self.null_statistic[self.nactive:]




class greedy_score_step_approx(greedy_score_step):

    def __init__(self, loss,
                 penalty,
                 active_groups,
                 inactive_groups,
                 randomization,
                 randomizer):

        greedy_score_step.__init__(self, loss,
                                 penalty,
                                 active_groups,
                                 inactive_groups,
                                 randomization)
        self.randomizer = randomizer


    def solve_approx(self):

        #self.solve()
        #self.setup_sampler()
        #p = self.inactive.sum()
        #print(self.selection_variable['variables'])
        #self._overall[self.selection_variable['variables']] = 1

        X, y = self.loss.data
        n, p = X.shape
        omega = self.randomization.sample()
        randomized_score = np.dot(X.T,y)+omega
        maximizing_var = np.argmax(np.abs(randomized_score))
        self._overall = np.zeros(X.shape[1], dtype=bool)
        self._overall[maximizing_var] = 1
        self.sign = np.sign(randomized_score[maximizing_var])
        self.observed_scaling = np.abs(randomized_score[maximizing_var])
        self.observed_subgradients = randomized_score[~self._overall]
        self._opt_linear_term = np.identity(p)
        self._opt_linear_term[0,0] = self.sign

        #self.observed_opt_state = np.hstack([self.observed_scaling, self.observed_subgradients])
        #_opt_linear_term = np.concatenate((np.atleast_2d(self.maximizing_subgrad).T, self.losing_padding_map), 1)
        #self._opt_linear_term = np.concatenate((_opt_linear_term[self._overall,:], _opt_linear_term[~self._overall,:]), 0)

        self.opt_transform = (self._opt_linear_term, np.zeros(p))


        self._score_linear_term = np.zeros((p,p))
        nactive = self._overall.sum()
        X, y = self.loss.data
        XE = X[:, self._overall]
        self._score_linear_term[:nactive, :nactive] = -np.dot(XE.T, XE)
        self._score_linear_term[nactive:, :nactive] = -np.dot(X[:,~self._overall].T, XE)
        self._score_linear_term[nactive:, nactive:] = -np.identity(X.shape[1]-nactive)

        from selection.randomized.M_estimator import restricted_Mest
        beta_unpenalized = restricted_Mest(self.loss, self._overall, solve_args={'min_its':50, 'tol':1.e-10})
        beta_full = np.zeros(self.loss.shape)
        beta_full[self._overall] = beta_unpenalized
        self.observed_score_state = np.hstack([beta_unpenalized,
                                               -self.loss.smooth_objective(beta_full, 'grad')[~self._overall]])
        self.inactive_lagrange = self.observed_scaling * self.penalty.weights[0] * np.ones(p-1)

        bootstrap_score = pairs_bootstrap_glm(self.loss,
                                              self.active,
                                              inactive=~self.active)[0]
        n, p = X.shape
        self.p = p
        score_cov = bootstrap_cov(lambda: np.random.choice(n, size=(n,), replace=True), bootstrap_score)
        self.score_target_cov = score_cov[:, :nactive]
        self.target_cov = score_cov[:nactive, :nactive]
        self.target_observed = self.observed_score_state[:nactive]


        self.nactive = nactive
        self.B_active = self._opt_linear_term[:nactive, :nactive]
        self.B_inactive = self._opt_linear_term[nactive:, :nactive]
        self.feasible_point = self.observed_scaling


    def setup_map(self, j):

        self.A = np.dot(self._score_linear_term, self.score_target_cov[:, j]) / self.target_cov[j, j]
        self.null_statistic = self._score_linear_term.dot(self.observed_score_state) - self.A * self.target_observed[j]

        self.offset_active = self.null_statistic[:self.nactive]
        self.offset_inactive = self.null_statistic[self.nactive:]


    def bootstrap_sample(self, solve_args={'min_its': 20, 'tol': 1.e-10}):
        X, y = self.loss.data
        n = X.shape[0]
        sampler = lambda: np.random.choice(n, size=(n,), replace=True)
        indices = sampler()
        _boot_loss = self.loss.subsample(indices)
        _beta_unpenalized = restricted_Mest(_boot_loss, self._overall)
        return _beta_unpenalized



















