import glob
import pandas as pd
import selection.tests.reports as reports
from selection.randomized.tests.test_power import compute_power

def read_results():
    path = "/Users/Jelena/high_dim_results/experiment_1/*.pkl"
    allFiles = glob.glob(path)

    list_dfs = []
    nfiles = 0
    for _file in allFiles:
        nfiles = nfiles+1
        df = pd.read_pickle(_file)
        list_dfs.append(df)

    print("nfiles", nfiles)
    dfs = pd.concat(list_dfs)
    return dfs

if __name__ == '__main__':
    dfs = read_results()

    compute_power(dfs)
    reports.summarize_all(dfs)
    fig = reports.pivot_plot_plus_naive(dfs)
    fig.suptitle('Randomized Lasso marginalized subgradient')
    fig.savefig('marginalized_subgrad_pivots.pdf')




