import numpy as np
import regreg.api as rr
from selection.bayesian.selection_probability_rr import nonnegative_softmax_scaled
from scipy.stats import norm

def myround(a, decimals=1):
    a_x = np.round(a, decimals=1)* 10.
    rem = np.zeros(a.shape[0], bool)
    rem[(np.remainder(a_x, 2) == 1)] = 1
    a_x[rem] = a_x[rem] + 1.
    return a_x/10.


class neg_log_cube_probability_laplace(rr.smooth_atom):
    def __init__(self,
                 q, #equals p - E in our case
                 lagrange,
                 randomization_scale = 1., #equals the randomization variance in our case
                 coef=1.,
                 offset=None,
                 quadratic=None):

        self.b = randomization_scale
        self.lagrange = lagrange
        self.q = q

        rr.smooth_atom.__init__(self,
                                (self.q,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=None,
                                coef=coef)

    def smooth_objective(self, arg, mode='both', check_feasibility=False, tol=1.e-6):

        arg = self.apply_offset(arg)

        arg_u = (arg + self.lagrange)/self.b
        arg_l = (arg - self.lagrange)/self.b
        scaled_lagrange = (2* self.lagrange)/self.b

        ind_arg_1 = np.zeros(self.q, bool)
        ind_arg_1[(arg_u <0.)] = 1
        ind_arg_2 = np.zeros(self.q, bool)
        ind_arg_2[(arg_l >0.)] = 1
        ind_arg_3 = np.logical_and(~ind_arg_1, ~ind_arg_2)
        cube_prob = np.zeros(self.q)
        cube_prob[ind_arg_1] = np.exp(arg_u[ind_arg_1])/2. - np.exp(arg_l[ind_arg_1])/2.
        cube_prob[ind_arg_2] = -np.exp(-arg_u[ind_arg_2])/2. + np.exp(-arg_l[ind_arg_2])/2.
        cube_prob[ind_arg_3] = 1- np.exp(-arg_u[ind_arg_3])/2. - np.exp(arg_l[ind_arg_3])/2.
        neg_log_cube_prob = -np.log(cube_prob).sum()

        log_cube_grad = np.zeros(self.q)
        log_cube_grad[ind_arg_1] = 1./self.b
        log_cube_grad[ind_arg_2] = np.true_divide((np.exp(-scaled_lagrange[ind_arg_2])+ 1.)/self.b,
                                                  np.exp(-scaled_lagrange[ind_arg_2])-1.)
        num_cube_grad = np.true_divide(np.exp(-scaled_lagrange[ind_arg_3]), 2 * self.b) - \
                        np.true_divide(np.exp((2* arg_l[ind_arg_3])), 2 * self.b)
        den_cube_grad = np.exp(arg_l[ind_arg_3]) - np.exp(-scaled_lagrange[ind_arg_3])/2. - \
                        np.exp(2* arg_l[ind_arg_3])/2.
        log_cube_grad[ind_arg_3] = np.true_divide(num_cube_grad,den_cube_grad)
        neg_log_cube_grad = -log_cube_grad

        if mode == 'func':
            return self.scale(neg_log_cube_prob)
        elif mode == 'grad':
            return self.scale(neg_log_cube_grad)
        elif mode == 'both':
            return self.scale(neg_log_cube_prob), self.scale(neg_log_cube_grad)
        else:
            raise ValueError("mode incorrectly specified")


class neg_log_cube_probability(rr.smooth_atom):
    def __init__(self,
                 q, #equals p - E in our case
                 lagrange,
                 randomization_scale = 1., #equals the randomization variance in our case
                 coef=1.,
                 offset=None,
                 quadratic=None):

        self.randomization_scale = randomization_scale
        self.lagrange = lagrange
        self.q = q

        rr.smooth_atom.__init__(self,
                                (self.q,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=None,
                                coef=coef)

    def smooth_objective(self, arg, mode='both', check_feasibility=False, tol=1.e-6):

        arg = self.apply_offset(arg)

        arg_u = (arg + self.lagrange)/self.randomization_scale
        arg_l = (arg - self.lagrange)/self.randomization_scale
        prod_arg = np.exp(-(2. * self.lagrange * arg)/(self.randomization_scale**2))
        neg_prod_arg = np.exp((2. * self.lagrange * arg)/(self.randomization_scale**2))
        cube_prob = norm.cdf(arg_u) - norm.cdf(arg_l)
        log_cube_prob = -np.log(cube_prob).sum()
        threshold = 10 ** -10
        indicator = np.zeros(self.q, bool)
        indicator[(cube_prob > threshold)] = 1
        positive_arg = np.zeros(self.q, bool)
        positive_arg[(arg>0)] = 1
        pos_index = np.logical_and(positive_arg, ~indicator)
        neg_index = np.logical_and(~positive_arg, ~indicator)
        log_cube_grad = np.zeros(self.q)
        log_cube_grad[indicator] = (np.true_divide(-norm.pdf(arg_u[indicator]) + norm.pdf(arg_l[indicator]),
                                        cube_prob[indicator]))/self.randomization_scale

        log_cube_grad[pos_index] = ((-1. + prod_arg[pos_index])/
                                     ((prod_arg[pos_index]/arg_u[pos_index])-
                                      (1./arg_l[pos_index])))/self.randomization_scale

        log_cube_grad[neg_index] = ((arg_u[neg_index] -(arg_l[neg_index]*neg_prod_arg[neg_index]))
                                    /self.randomization_scale)/(1.- neg_prod_arg[neg_index])


        if mode == 'func':
            return self.scale(log_cube_prob)
        elif mode == 'grad':
            return self.scale(log_cube_grad)
        elif mode == 'both':
            return self.scale(log_cube_prob), self.scale(log_cube_grad)
        else:
            raise ValueError("mode incorrectly specified")


class approximate_conditional_prob(rr.smooth_atom):

    def __init__(self,
                 t, #point at which density is to computed
                 map,
                 coef = 1.,
                 offset= None,
                 quadratic= None):

        self.t = t
        self.map = map
        self.q = map.p - map.nactive
        self.inactive_conjugate = self.active_conjugate = map.randomization.CGF_conjugate

        if self.active_conjugate is None:
            raise ValueError(
                'randomization must know its CGF_conjugate -- currently only isotropic_gaussian and laplace are implemented and are assumed to be randomization with IID coordinates')

        self.inactive_lagrange = self.map.inactive_lagrange

        rr.smooth_atom.__init__(self,
                                (map.nactive,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=self.map.feasible_point,
                                coef=coef)

        self.coefs[:] = map.feasible_point

        self.nonnegative_barrier = nonnegative_softmax_scaled(self.map.nactive)


    def sel_prob_smooth_objective(self, param, mode='both', check_feasibility=False):

        param = self.apply_offset(param)

        data = np.squeeze(self.t *  self.map.A)

        offset_active = self.map.offset_active + data[:self.map.nactive]
        offset_inactive = self.map.offset_inactive + data[self.map.nactive:]

        active_conj_loss = rr.affine_smooth(self.active_conjugate,
                                            rr.affine_transform(self.map.B_active, offset_active))

        if self.map.randomizer == 'laplace':
            cube_obj = neg_log_cube_probability_laplace(self.q, self.inactive_lagrange, randomization_scale = 1.)
        elif self.map.randomizer == 'gaussian':
            cube_obj = neg_log_cube_probability(self.q, self.inactive_lagrange, randomization_scale = 1.)

        cube_loss = rr.affine_smooth(cube_obj, rr.affine_transform(self.map.B_inactive, offset_inactive))

        total_loss = rr.smooth_sum([active_conj_loss,
                                    cube_loss,
                                    self.nonnegative_barrier])

        if mode == 'func':
            f = total_loss.smooth_objective(param, 'func')
            return self.scale(f)
        elif mode == 'grad':
            g = total_loss.smooth_objective(param, 'grad')
            return self.scale(g)
        elif mode == 'both':
            f, g = total_loss.smooth_objective(param, 'both')
            return self.scale(f), self.scale(g)
        else:
            raise ValueError("mode incorrectly specified")

    def minimize2(self, step=1, nstep=30, tol=1.e-6):

        current = self.coefs
        current_value = np.inf

        objective = lambda u: self.sel_prob_smooth_objective(u, 'func')
        grad = lambda u: self.sel_prob_smooth_objective(u, 'grad')

        for itercount in range(nstep):
            newton_step = grad(current)

            # make sure proposal is feasible

            count = 0
            while True:
                count += 1
                proposal = current - step * newton_step
                #print("current proposal and grad", proposal, newton_step)
                if np.all(proposal > 0):
                    break
                step *= 0.5
                if count >= 40:
                    #print(proposal)
                    raise ValueError('not finding a feasible point')

            # make sure proposal is a descent

            count = 0
            while True:
                proposal = current - step * newton_step
                proposed_value = objective(proposal)
                #print(current_value, proposed_value, 'minimize')
                if proposed_value <= current_value:
                    break
                step *= 0.5

            # stop if relative decrease is small

            if np.fabs(current_value - proposed_value) < tol * np.fabs(current_value):
                current = proposal
                current_value = proposed_value
                break

            current = proposal
            current_value = proposed_value

            if itercount % 4 == 0:
                step *= 2

        # print('iter', itercount)
        value = objective(current)

        return current, value

class approximate_conditional_density(rr.smooth_atom):

    def __init__(self, sel_alg,
                       coef=1.,
                       offset=None,
                       quadratic=None,
                       nstep=10):

        self.sel_alg = sel_alg

        rr.smooth_atom.__init__(self,
                                (1,),
                                offset=offset,
                                quadratic=quadratic,
                                coef=coef)

        self.target_observed = self.sel_alg.target_observed
        self.nactive = self.target_observed.shape[0]
        self.target_cov = self.sel_alg.target_cov

    def solve_approx(self):


        self.ind_obs = np.zeros(self.nactive, int)
        pvalues = np.zeros(self.nactive)

        for j in range(self.nactive):
            pvalues[j] = self.approx_conditional_prob(j)
            print(pvalues[j])
        return pvalues

    def approx_conditional_prob(self, j, B=1000):

        self.sel_alg.setup_map(j)
        bootstrap_sample = np.zeros(B)
        approx_sel_probabilities = np.zeros(B)
        for i in range(B):
            bootstrap_sample[i] = self.sel_alg.bootstrap_sample(j)
            #print(bootstrap_sample[i]- self.target_observed[j])
            approx = approximate_conditional_prob(bootstrap_sample[i] - self.target_observed[j], self.sel_alg)
            approx_sel_probabilities[i] = -(approx.minimize2(j, nstep=100)[::-1])[0]
        valid = ~np.isnan(approx_sel_probabilities)

        pivot = np.sum(np.multiply(np.exp(approx_sel_probabilities[valid]),
                                   np.array(bootstrap_sample[valid] < 2 * self.target_observed[j], dtype=int)))

        pivot = pivot / np.sum(np.exp(approx_sel_probabilities[valid]))

        return 2*min(pivot, 1-pivot)

