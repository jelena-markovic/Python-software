import numpy as np
from selection.sampling.randomized.tests.test_lasso_fixedX_saturated import selection
from selection.sampling.randomized.tests.test_lasso_fixedX_saturated import test_lasso

from selection.sampling.randomized.intervals.estimation import estimation, instance


class intervals():

    def __init__(self, X, y, active, betaE, cube, epsilon, lam, sigma, tau):

        (self.X, self.y,
         self.active,
         self.betaE, self.cube,
         self.epsilon,
         self.lam,
         self.sigma,
         self.tau) = (X, y,
                      active,
                      betaE, cube,
                      epsilon,
                      lam,
                      sigma,
                      tau)

        self.sigma_sq = self.sigma ** 2
        self.n, self.p = X.shape
        self.nactive = np.sum(active)
        self.ninactive = self.p - self.nactive
        self.XE_pinv = np.linalg.pinv(self.X[:, self.active])

        self.eta_norm_sq = np.zeros(self.nactive)
        for j in range(self.nactive):
            eta = self.XE_pinv[j, :]
            self.eta_norm_sq[j] = np.linalg.norm(eta) ** 2

        self.grid_length = 400
        self.param_values_at_grid = np.linspace(-10, 10, num=self.grid_length)



    def setup_samples(self, ref_vec, samples, observed, variances):
        (self.ref_vec,
         self.samples,
         self.observed,
         self.variances) = (ref_vec,
                            samples,
                            observed,
                            variances)

        self.nsamples = self.samples.shape[1]


    def empirical_exp(self, j, param):
        ref = self.ref_vec[j]
        factor = np.true_divide(param-ref, self.eta_norm_sq[j]*self.sigma_sq)
        tilted_samples = np.exp(self.samples[j,:]*factor)
        return np.sum(tilted_samples)/float(self.nsamples)


    def pvalue_by_tilting(self, j, param):
        ref = self.ref_vec[j]
        indicator = np.array(self.samples[j,:] < self.observed[j], dtype =int)
        log_gaussian_tilt = np.array(self.samples[j,:]) * (param - ref)
        log_gaussian_tilt /= self.eta_norm_sq[j]*self.sigma_sq
        emp_exp = self.empirical_exp(j, param)
        LR = np.true_divide(np.exp(log_gaussian_tilt), emp_exp)
        return np.clip(np.sum(np.multiply(indicator, LR)) / float(self.nsamples), 0, 1)


    def pvalues_param_all(self, param_vec):
        pvalues = []
        for j in range(self.nactive):
            pvalues.append(self.pvalue_by_tilting(j, param_vec[j]))
        return pvalues

    def pvalues_ref_all(self):
        pvalues = []
        for j in range(self.nactive):
            indicator = np.array(self.samples[j, :] < self.observed[j], dtype=int)
            pvalues.append(np.sum(indicator)/float(self.nsamples))
        return pvalues


    def pvalues_grid(self, j):
        pvalues_at_grid = [self.pvalue_by_tilting(j, self.param_values_at_grid[i]) for i in range(self.grid_length)]
        pvalues_at_grid = np.asarray(pvalues_at_grid, dtype=np.float32)
        return pvalues_at_grid


    def construct_intervals(self, j, alpha=0.1):
        pvalues_at_grid = self.pvalues_grid(j)
        accepted_indices = np.array(pvalues_at_grid > alpha)
        if np.sum(accepted_indices)>0:
            self.L = np.min(self.param_values_at_grid[accepted_indices])
            self.U = np.max(self.param_values_at_grid[accepted_indices])
            return self.L, self.U

    def construct_intervals_all(self, truth_vec, alpha=0.1):
        ncovered = 0
        nparam = 0
        for j in range(self.nactive):
            LU = self.construct_intervals(j, alpha=alpha)
            if LU is not None:
                L, U = LU
                print "interval", L, U
                nparam +=1
                if (L <= truth_vec[j]) and (U >= truth_vec[j]):
                     ncovered +=1
        return ncovered, nparam



def test_intervals(n=100, p=10, s=3, reference="selective MLE"):

    tau = 1.
    data_instance = instance(n, p, s)
    X, y, true_beta, nonzero, sigma = data_instance.generate_response()
    y0 = y.copy()
    random_Z = np.random.standard_normal(p)
    lam, epsilon, active, betaE, cube, initial_soln = selection(X,y, random_Z)

    if lam < 0:
        return None
    int_class = intervals(X, y, active, betaE, cube, epsilon, lam, sigma, tau)
    nactive = np.sum(active)

    if reference=="selective MLE":
        est = estimation(X, y, active, betaE, cube, epsilon, lam, sigma, tau)
        ref_vec = est.compute_mle_all()
    else:
        if reference=="unpenalized MLE":
            ref_vec = np.dot(int_class.XE_pinv.T, y0)
        else:
            raise ValueError("Wrong reference")

    param_vec = true_beta[active]
    print "true vector", param_vec
    print "reference", ref_vec

    # running the Langevin sampler
    _, _, all_observed, all_variances, all_samples = test_lasso(X, y, nonzero, sigma, lam, epsilon, active, betaE,
                                                                cube, random_Z, beta_reference=ref_vec.copy(),
                                                                randomization_distribution="normal",
                                                                Langevin_steps=20000, burning=2000)


    int_class.setup_samples(ref_vec.copy(), all_samples, all_observed, all_variances)

    pvalues_ref = int_class.pvalues_ref_all()
    pvalues_param = int_class.pvalues_param_all(param_vec)

    #pvalues_param = int_class.pvalue_by_tilting(0, param_vec[0])
    ncovered, nparam = 0, 0
    #coverage, nparam = int_class.construct_intervals_all(param_vec)

    print "pvalue(s) at the truth", pvalues_param
    return np.copy(pvalues_ref), np.copy(pvalues_param), ncovered, nparam



if __name__ == "__main__":

    P_param_all = []
    P_ref_all = []
    P_param_first = []
    P_ref_first = []
    ncovered_total = 0
    nparams_total = 0

    for i in range(50):
        print "\n"
        print "iteration", i
        pvals_ints = test_intervals()
        if pvals_ints is not None:
            #print pvalues
            pvalues_ref, pvalues_param, ncovered, nparam = pvals_ints
            P_ref_all.extend(pvalues_ref)
            P_param_all.extend(pvalues_param)
            P_ref_first.append(pvalues_ref[0])
            P_param_first.append(pvalues_param[0])

            if ncovered is not None:
                ncovered_total += ncovered
                nparams_total += nparam


    #print "number of intervals", nparams
    #print "coverage", ncovered/float(nparams)



    from matplotlib import pyplot as plt
    import statsmodels.api as sm


    fig = plt.figure()
    plot_pvalues0 = fig.add_subplot(221)
    plot_pvalues1 = fig.add_subplot(222)
    plot_pvalues2 = fig.add_subplot(223)
    plot_pvalues3 = fig.add_subplot(224)


    ecdf = sm.distributions.ECDF(P_param_all)
    x = np.linspace(min(P_param_all), max(P_param_all))
    y = ecdf(x)
    plot_pvalues0.plot(x, y, '-o', lw=2)
    plot_pvalues0.plot([0, 1], [0, 1], 'k-', lw=2)
    plot_pvalues0.set_title("P values at the truth")
    plot_pvalues0.set_xlim([0, 1])
    plot_pvalues0.set_ylim([0, 1])

    ecdf1 = sm.distributions.ECDF(P_ref_all)
    x1 = np.linspace(min(P_ref_all), max(P_ref_all))
    y1 = ecdf1(x1)
    plot_pvalues1.plot(x1, y1, '-o', lw=2)
    plot_pvalues1.plot([0, 1], [0, 1], 'k-', lw=2)
    plot_pvalues1.set_title("P values at the reference")
    plot_pvalues1.set_xlim([0, 1])
    plot_pvalues1.set_ylim([0, 1])


    ecdf2 = sm.distributions.ECDF(P_ref_first)
    x2 = np.linspace(min(P_ref_first), max(P_ref_first))
    y2 = ecdf2(x2)
    plot_pvalues2.plot(x2, y2, '-o', lw=2)
    plot_pvalues2.plot([0, 1], [0, 1], 'k-', lw=2)
    plot_pvalues2.set_title("First P value at the reference")
    plot_pvalues2.set_xlim([0, 1])
    plot_pvalues2.set_ylim([0, 1])

    ecdf3 = sm.distributions.ECDF(P_param_first)
    x3 = np.linspace(min(P_param_first), max(P_param_first))
    y3 = ecdf3(x3)
    plot_pvalues3.plot(x3, y3, '-o', lw=2)
    plot_pvalues3.plot([0, 1], [0, 1], 'k-', lw=2)
    plot_pvalues3.set_title("First P value at the truth")
    plot_pvalues3.set_xlim([0, 1])
    plot_pvalues3.set_ylim([0, 1])

    plt.show()
    plt.savefig("P values from the intervals file.png")
