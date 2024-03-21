import cProfile
import pstats

import pandas as pd
import numpy as np

from optiver_trading_at_the_close.feature_engineering import FE
# from optiver_trading_at_the_close.nearest_neightbors_features import NearestNeighborsFeatures
from optiver_trading_at_the_close.test import NearestNeighborsFeatures


if __name__ == '__main__':
    DATA_PATH = './../data/train.csv'
    
    df = pd.read_csv(DATA_PATH)
    
    df = df.dropna(subset=['target'], axis=0)

    X_train = df.loc[df['date_id'] <= 420]
    X_test = df.loc[df['date_id'] > 420]

    y_train = X_train['target']
    X_train = X_train.drop(columns='target')

    y_test = X_test['target']
    X_test = X_test.drop(columns='target')
    
    aux = X_train.copy()

    fe = FE()

    aux = fe.fit_transform(aux)
    
    nnf = NearestNeighborsFeatures(
        features_to_use_for_distance_computation=['seconds_in_bucket', 'wap', 'bid_plus_ask_sizes', 'bid_ask_size_imb'],
        features_get=['wap', 'bid_plus_ask_sizes', 'bid_ask_size_imb'],
        metrics=['l1', 'l2'],
        n_jobs=-1
    )

    nnf.fit(aux, y_train)
    
    print('START PROFILE')
    pr = cProfile.Profile()
    pr.enable()
    aux_transform = nnf.transform(aux)
    pr.disable()
    
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats('./l1_l2_metric_profile_new_join.prof')
        
    print('ENDDD')