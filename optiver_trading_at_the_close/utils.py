import pandas as pd
import numpy as np

def time_corr_feature_importance(X, target, date, corr='spearman'):
    results = X.groupby(date, group_keys=False, sort=False).corr(corr)
    results2 = results.reset_index().groupby('index', group_keys=False, sort=False)[target].agg(['mean', 'std'])

    results2['standard_deviation'] = results2['mean']/results2['std']

    results2['abs_mean'] = np.abs(results2['mean'])
    results2['abs_standard_deviation'] = np.abs(results2['standard_deviation'])

    return results2.sort_values(by='abs_standard_deviation', ascending=False)