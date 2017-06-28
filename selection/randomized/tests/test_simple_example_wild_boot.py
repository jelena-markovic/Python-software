import numpy as np
from scipy.stats import laplace, probplot, uniform

from matplotlib import pyplot as plt
from selection.sampling.langevin import projected_langevin
from selection.distributions.discrete_family import discrete_family
import statsmodels.api as sm


def test_simple_problem(noise = "normal", n=100, randomization_dist = "logistic", threshold =1,
                        weights="neutral",
                        Langevin_steps=8000, burning = 1000):

    step_size = 10./n
    truth = -1./np.sqrt(n)
    if noise == "normal":
        y = np.random.standard_normal(n) + truth
    elif noise=="laplace":
        y = np.random.laplace(loc=0, scale=1. / np.sqrt(2), size=n) + truth
    elif noise == "uniform":
        y = np.random.uniform(low=-np.sqrt(3), high=np.sqrt(3), size=n)+truth
    elif noise == "logistic":
        y = np.random.logistic(loc=0, scale=np.sqrt(3) / np.pi, size=n)+truth


    obs = np.sqrt(n)*np.mean(y)

    if randomization_dist=="logistic":
        omega = np.random.logistic(loc=0, scale=1, size=1)

    if (obs+omega<threshold):
        return -1

    #initial_state = np.ones(n)
    initial_state = np.zeros(n)

    y_cs = (y-np.mean(y))/np.sqrt(n)

    def full_projection(state):
        return state

    def full_gradient(state, n=n, y_cs = y_cs):

        gradient = np.zeros(n)
        if weights == "normal":
            gradient -= state
        if (weights == "gumbel"):
            gumbel_beta = np.sqrt(6) / (1.14 * np.pi)
            euler = 0.57721
            gumbel_mu = -gumbel_beta * euler
            gumbel_sigma = 1. / 1.14
            gradient -= (1. - np.exp(-(state * gumbel_sigma - gumbel_mu) / gumbel_beta)) * gumbel_sigma / gumbel_beta
        if weights == "logistic":
            gradient = np.divide(np.exp(-state)-1, np.exp(-state)+1)

        if weights =="neutral":
            gradient = - np.inner(state, y_cs) * y_cs

        omega = -np.sqrt(n)*truth-np.inner(y_cs, state) + threshold
        if randomization_dist=="logistic":
            randomization_derivative = -1./(1+np.exp(-omega)) # derivative of log\bar{G}(omega) wrt omega

        gradient -= y_cs * randomization_derivative

        return gradient


    sampler = projected_langevin(initial_state.copy(),
                                 full_gradient,
                                 full_projection,
                                 step_size)

    samples = []

    for i in range(Langevin_steps):
        sampler.next()
        if (i > burning):
            samples.append(sampler.state.copy())

    alphas = np.array(samples)

    pop = [np.inner(y_cs, alphas[i,:]) for i in range(alphas.shape[0])]

    fam = discrete_family(pop, np.ones_like(pop))
    obs -= np.sqrt(n)*truth
    pval = fam.cdf(0, obs)
    pval = 2 * min(pval, 1 - pval)
    print "observed: ", obs, "p value: ", pval
    return pval


if __name__ == "__main__":

    np.random.seed(1)
    fig = plt.figure()
    fig.suptitle('Pivots for the simple example wild bootstrap')

    for noise in ['normal', 'laplace', 'uniform', 'logistic']:
        P = []
        for i in range(1000):
            pval = test_simple_problem(noise=noise)
            if pval>-1:
                print i
                P.append(pval)
        print noise

        # generates one plot for the p-values for all types of errors
        if (noise == 'normal'):
            plot1 = fig.add_subplot(221)
            plot1.set_title('Normal errors')
        if (noise == 'laplace'):
            plot1 = fig.add_subplot(222)
            plot1.set_title('Laplace errors')
        if (noise == 'uniform'):
            plot1 = fig.add_subplot(223)
            plot1.set_title('Uniform errors')
        if (noise == 'logistic'):
            plot1 = fig.add_subplot(224)
            plot1.set_title('Logistic errors')

        ecdf = sm.distributions.ECDF(P)
        x = np.linspace(min(P), max(P))
        y = ecdf(x)
        plt.plot(x, y, '-o', lw=2)
        plt.plot([0, 1], [0, 1], 'k-', lw=1)
    # plt.savefig('foo.pdf')
    plt.show()
