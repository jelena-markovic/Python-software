from rpy2.robjects.packages import importr
from rpy2 import robjects
knockoff = importr('knockoff')
from selection.tests.instance import gaussian_instance
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import numpy as np
from selection.tests.flags import SET_SEED, SMALL_SAMPLES
from selection.tests.decorators import (wait_for_return_value,
                                        set_seed_iftrue,
                                        set_sampling_params_iftrue)


@set_sampling_params_iftrue(SMALL_SAMPLES, ndraw=10, burnin=10)
@set_seed_iftrue(SET_SEED)
@wait_for_return_value()
def test_knockoffs(s=30,
                   n=3000,
                   p=1000,
                   rho=0.6,
                   equi_correlated=False,
                   snr=3.5):

    X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=rho, snr=snr, sigma=1.,
                                                       equi_correlated=equi_correlated)

    robjects.r('''knockoffs = function(X,y){
        q=0.2
        filter = knockoff.filter(X, y, fdr=q, threshold ="knockoff")
        selected = filter$selected
        return(selected)
        }''')
    knockoffs = robjects.globalenv['knockoffs']

    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_y = robjects.r.matrix(y, nrow=n, ncol=1)
    selection = np.array(knockoffs(r_X, r_y))
    print(selection)
    selection= selection-1
    decisions = np.zeros(p)
    #for j in range(len(decisions)):
    decisions[selection] = 1

    TP = np.sum(decisions[nonzero])
    R = sum(decisions)
    FP = R-TP
    FDR = (R-TP)/float(max(1,R))
    power = TP/s

    return FDR, power, FP, R


def compute_power(niters=50, **kwargs):
    FDR_sample, power_sample, rejections_sample, FP_sample = [], [], [], []
    for i in range(niters):
        print("iteration", i)
        result = test_knockoffs(**kwargs)[1]
        if result is not None:
            FDR, power, FP, rejections = result
            FDR_sample.append(FDR)
            power_sample.append(power)
            FP_sample.append(FP)
            rejections_sample.append(rejections)

            print("FDP knockoffs mean", np.mean([i for i in FDR_sample]))
            print("power knockoffs mean", np.mean([i for i in power_sample]))
            print("false rejections knockoffs", np.mean([i for i in FP_sample]))
            print("total rejections knockoffs", np.mean([i for i in rejections_sample]))

    return None

if __name__ == '__main__':
    np.random.seed(500)
    kwargs = {'s':30, 'n':3000, 'p':1000, 'rho':0.6,
              'equi_correlated':False, 'snr':3.5}
    compute_power(**kwargs)




