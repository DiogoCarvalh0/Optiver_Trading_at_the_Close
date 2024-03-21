from itertools import combinations
import warnings

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
warnings.simplefilter(action='ignore')


class FE(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self, X, y=None):
        self.dtypes_map = X.dtypes
        
        agg = X.groupby('stock_id').agg({
            'bid_size': ['median', 'std', 'max', 'min'],
            'ask_size': ['median', 'std'],
            'bid_price': ['median', 'std', 'max', 'min'],
            'ask_price': ['median', 'std'],
        })
        # agg_first_1_mins = X[X['seconds_in_bucket']<=60].groupby('stock_id').agg({'bid_size': ['median', 'std'], 'ask_size': ['median', 'std']})
        # agg_first_3_mins = X[X['seconds_in_bucket']<=180].groupby('stock_id').agg({'bid_size': ['median', 'std'], 'ask_size': ['median', 'std']})
        # agg_first_5_mins = X[X['seconds_in_bucket']<=300].groupby('stock_id').agg({'bid_size': ['median', 'std'], 'ask_size': ['median', 'std']})

        self.stocks_info = {
            'median_size': agg[('bid_size', 'median')] + agg[('ask_size', 'median')],
            'std_size': agg[('bid_size', 'std')] + agg[('ask_size', 'std')],
            'ptp_size': agg[('bid_size', 'max')] - agg[('bid_size', 'min')],
            'median_liquidity_imbalance': (agg[('bid_size', 'median')] - agg[('ask_size', 'median')])/(agg[('bid_size', 'median')] + agg[('ask_size', 'median')]),
            
            'median_price': agg[('bid_price', 'median')] + agg[('ask_price', 'median')],
            'std_price': agg[('bid_price', 'std')] + agg[('ask_price', 'std')],
            'ptp_price': agg[('bid_price', 'max')] - agg[('bid_price', 'min')],
            'median_liquidity_imbalance_price': (agg[('bid_price', 'median')] - agg[('ask_price', 'median')])/(agg[('bid_price', 'median')] + agg[('ask_price', 'median')]),
            
            'median_gain_market_value': (agg[('ask_price', 'median')] * agg[('ask_size', 'median')]) - (agg[('bid_price', 'median')] * agg[('bid_size', 'median')])
            # 'median_size_first_1_mins': agg_first_1_mins[('bid_size', 'median')] + agg_first_1_mins[('ask_size', 'median')],
            # 'std_size_first_1_mins': agg_first_1_mins[('bid_size', 'std')] + agg_first_1_mins[('ask_size', 'std')],
            
            # 'median_size_first_3_mins': agg_first_3_mins[('bid_size', 'median')] + agg_first_3_mins[('ask_size', 'median')],
            # 'std_size_first_3_mins': agg_first_3_mins[('bid_size', 'std')] + agg_first_3_mins[('ask_size', 'std')],
            
            # 'median_size_first_5_mins': agg_first_5_mins[('bid_size', 'median')] + agg_first_5_mins[('ask_size', 'median')],
            # 'std_size_first_5_mins': agg_first_5_mins[('bid_size', 'std')] + agg_first_5_mins[('ask_size', 'std')],
        }
        
        self.is_fit_ = True
        
        return self
        
    def transform(self, X):
        check_is_fitted(self, 'is_fit_')
        
        interset_cols = list(set(X.columns).intersection(set(self.dtypes_map.keys())))
        
        _X = X.copy()
        _X[interset_cols] = _X[interset_cols].astype(self.dtypes_map[interset_cols], errors='ignore')
        _X = _X.round(6)
        
        _X['imbalance_ratio'] = _X['imbalance_size'] / _X['matched_size']
        _X['imbalance_size_flag'] = _X['imbalance_size'] * _X['imbalance_buy_sell_flag']
        _X['size_spread'] = _X['bid_size'] - _X['ask_size']
        _X['bid_plus_ask_sizes'] = (_X['bid_size'] + _X['ask_size']).astype(np.float32)
        _X['mid_price'] = ((_X['ask_price'] + _X['bid_price']) / 2).astype(np.float32)
        _X['size_imbalance'] = _X['bid_size'] / _X['ask_size']
        _X['imb_mat_over_bid_ask_size'] = (_X['imbalance_size']-_X['matched_size'])/(_X['bid_size']-_X['ask_size'])
        _X['liquidity_imbalance'] = (_X['bid_size']-_X['ask_size'])/(_X['bid_size']+_X['ask_size'])
        
        _X['imbalance_size_over_bid_size'] = _X['imbalance_size']/_X['bid_size']
        _X['imbalance_size_over_ask_size'] = _X['imbalance_size']/_X['ask_size']
        _X['matched_size_over_ask_size'] = _X['matched_size']/_X['bid_plus_ask_sizes']
        
        # get train averages
        _X = self.get_median_std_values(_X)
        
        # get time statistics
        _X = self.get_time_statistics(_X)
        
        _X['bid_size_index_bid_size_ratio'] = _X['bid_size']/_X['mean_index_bid_size']
        _X['ask_size_index_ask_size_ratio'] = _X['ask_size']/_X['mean_index_ask_size']
        _X['bid_price_index_bid_price_ratio'] = _X['bid_price']/_X['mean_index_bid_price']
        _X['ask_price_index_ask_price_ratio'] = _X['ask_price']/_X['mean_index_ask_price']
        
        _X['coeficient_variance_index_wap_raw'] = _X['std_index_wap_raw']/_X['mean_index_wap_raw']*100
        
        _X['mean_index_wap'] = (_X['mean_index_bid_price']*_X['mean_index_ask_size'] + _X['mean_index_ask_price']*_X['mean_index_bid_size'])/(_X['mean_index_bid_size'] + _X['mean_index_ask_size'])
        _X['wap_over_mean_index_wap'] = _X['wap']/_X['mean_index_wap']
        
        # more feats
        _X["price_spread"] = (_X["ask_price"] - _X["bid_price"]).astype(np.float32)
        _X['gain_market_value'] = (_X['ask_price']*_X['ask_size']) - (_X['bid_price']*_X['bid_size'])
        gb = _X.groupby(['stock_id', 'date_id'], group_keys=False, sort=False)
        _X["imbalance_momentum"] = gb['imbalance_size'].diff() / _X['matched_size']
        _X["spread_intensity"] = gb['price_spread'].diff()
        _X['price_pressure'] = _X['imbalance_size'] * _X["price_spread"]
        _X['market_urgency'] = _X['price_spread'] * _X['liquidity_imbalance']
        _X['depth_pressure'] = _X["price_spread"] * (_X['far_price'] - _X['near_price'])
        
        _X['mean_index_size_spread'] = _X["mean_index_ask_size"] - _X["mean_index_bid_size"]
        _X['mean_index_mid_price'] = (_X["mean_index_ask_price"] + _X["mean_index_bid_price"])/2
        
        _X['matched_pressure'] = _X['matched_size']/_X['size_spread']
        _X['imbalance_pressure'] = _X['imbalance_size']/_X['size_spread']
        
        _X['mean_index_sum_size'] = _X["mean_index_ask_size"] + _X["mean_index_bid_size"]
        
        # weights features
        _X = self.get_stock_weights(_X)
        _X["weighted_wap"] = _X["weights"] * _X["wap"]
        _X['wap_momentum'] = _X.groupby('stock_id')['weighted_wap'].pct_change(periods=6)
        
        _X['spread_depth_ratio'] = _X['price_spread'] / (_X['bid_size'] + _X['ask_size'])
        _X['mid_price_movement'] = _X['mid_price'].diff(periods=5).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    
        _X['micro_price'] = ((_X['bid_price'] * _X['ask_size']) + (_X['ask_price'] * _X['bid_size'])) / (_X['bid_size'] + _X['ask_size'])
        _X['relative_spread'] = (_X['ask_price'] - _X['bid_price']) / _X['wap']
        
        _X['mid_price_times_volume'] = _X['mid_price_movement'] * _X['bid_plus_ask_sizes']
        _X['harmonic_imbalance'] = 2/ ((1/_X['bid_size']) + (1/_X['ask_size'])) 
        
        # imbalance features
        pair_relationships = [
            ['reference_price', 'wap'],
            ['reference_price', 'bid_price'],
            ['reference_price', 'ask_price'],
            
            ['near_price', 'wap'],
            ['near_price', 'bid_price'],
            ['near_price', 'ask_price'],
            
            ['far_price', 'wap'],
            ['far_price', 'bid_price'],
            ['far_price', 'ask_price'],
            
            ['wap', 'bid_price'],
            ['wap', 'ask_price'],
            ['wap', 'mid_price'],
            ['wap', 'weighted_wap'],
            ['wap', 'mean_index_wap'],
            ['wap', 'mean_index_mid_price'],
            
            ['mean_index_bid_price', 'mean_index_wap'],
            ['mean_index_ask_price', 'mean_index_wap'],
            ['mean_index_mid_price', 'mean_index_wap'],
            ['mean_index_wap_raw', 'mean_index_wap'],
            
            ['matched_size', 'median_size'],
            ['matched_size', 'ask_size'],
            ['matched_size', 'bid_size'],
            ['matched_size', 'size_spread'],
            
            ['imbalance_size', 'median_size'],
            ['imbalance_size', 'ask_size'],
            ['imbalance_size', 'bid_size'],
            ['imbalance_size', 'size_spread'],
            
            ['imbalance_size', 'matched_size'],
            
            ['median_size', 'mean_index_bid_size'],
            ['median_size', 'mean_index_ask_size'],
            ['median_size', 'mean_index_sum_size'],
            ['median_size', 'norm_sum_size'],
            
            ['mean_index_ask_size', 'mean_index_bid_size'],
            ['mean_index_ask_size', 'norm_ask_size'],
            ['mean_index_bid_size', 'norm_bid_size'],
            ['norm_ask_size', 'norm_bid_size'],
            ['bid_size', 'norm_bid_size'],
            ['ask_size', 'norm_ask_size'],
            ['matched_size', 'norm_bid_size'],
            ['matched_size', 'norm_ask_size'],
            ['matched_size', 'norm_sum_size'],
            ['matched_size', 'norm_size_spread'],
            ['norm_sum_size', 'norm_size_spread'],
            
            ['size_spread', 'mean_index_size_spread'],
            
            # ['market_urgency', 'gain_market_value'],
            # ['depth_pressure', 'gain_market_value'],
            ['market_urgency', 'depth_pressure'],
            ['gain_market_value', 'median_gain_market_value']
        ]
        _X = self.pair_imbalance(_X, pair_relationships)
        triple_relationships = [
            ['wap', 'ask_price', 'bid_price'],
            ['reference_price', 'ask_price', 'bid_price'],
            ['reference_price', 'ask_price', 'wap'],
            ['reference_price', 'wap', 'bid_price'],
            ['reference_price', 'near_price', 'far_price'],
            
            ['near_price', 'mean_index_ask_price', 'mean_index_bid_price'],
            ['wap', 'mean_index_wap_raw', 'mean_index_wap'],
            ['wap', 'weighted_wap', 'mean_index_wap'],
            ['wap', 'mid_price', 'mean_index_wap'],
            ['wap', 'weighted_wap', 'mid_price'],
            ['market_urgency', 'depth_pressure', 'gain_market_value'],
            
            
            ["matched_size", "bid_size", "ask_size"],
            ["matched_size", "bid_size", "imbalance_size"],
            ["matched_size", "ask_size", "imbalance_size"],
            ["bid_size", "ask_size", "imbalance_size"],
            
            ['median_size', 'norm_bid_size', 'norm_ask_size'],
            ['median_size', 'norm_sum_size', 'norm_size_spread'],
            ['imbalance_size', 'matched_size', 'size_spread'],
            ['imbalance_size', 'matched_size', 'norm_sum_size']
        ]
        _X = self.triple_imbalance(_X, triple_relationships)
        
        # last day target features
        _X = self.past_day_target(_X)
        
        # rolling features
        rolling_feats = [
            'wap',
            'mean_index_wap',
            'imbalance_size_flag'
        ]
        single_window_rolling_feats = [
            'norm_sum_size_minus_norm_size_spread',
            'coeficient_variance_index_wap_raw',
            'wap_minus_mid_price',
            'bid_plus_ask_sizes',
            'near_price_mean_index_ask_price_mean_index_bid_price_imb2',
            'matched_size_median_size_imb',
            'reference_price_wap_imb',
            'imbalance_size_minus_matched_size'
        ]
        
        rolling_windows = [6, 24, 55]
        _X = self.rolling_features(_X, rolling_feats, windows=rolling_windows)
        
        _X = self.rolling_features(_X, single_window_rolling_feats, windows=[55])
        
        # RSI
        # _X = self.RSI(_X, rolling_feats, windows=rolling_windows)
        
        # Change features
        change_feats = rolling_feats + single_window_rolling_feats
        _X = self.change_features(_X, change_feats, windows=[1, 6, 12])
        _X = self.rolling_features(
            _X,
            [
                "wap_pct_change_1",
                "wap_pct_change_6",
                "mean_index_wap_pct_change_1",
                "mean_index_wap_pct_change_6",
            ],
            windows=rolling_windows
        )
        _X = self.pair_imbalance(_X, [
            ['historical_volatility_55_mean_index_wap_pct_change_6', 'historical_volatility_55_mean_index_wap_pct_change_1'],
            ['historical_volatility_55_wap_pct_change_6', 'historical_volatility_55_wap_pct_change_1'],
            ['historical_volatility_55_mean_index_wap_pct_change_6', 'historical_volatility_55_wap_pct_change_6'],
            ['historical_volatility_55_mean_index_wap_pct_change_1', 'historical_volatility_55_wap_pct_change_1'],
        ])
        
        
        # Diff features
        diff_feats = rolling_feats + single_window_rolling_feats
        _X = self.diff_features(_X, diff_feats, windows=[1, 6])
        
        # try to predict future target
        _X['future_wap'] = _X['wap'] + _X['wap']*_X["wap_pct_change_6"]
        _X['future_mean_index_wap'] = _X['mean_index_wap'] + _X['mean_index_wap']*_X["mean_index_wap_pct_change_6"]
        _X['estimated_target'] = ((_X['future_wap']/_X['wap']) - (_X['future_mean_index_wap']/_X['mean_index_wap'])) * 10000
        
        _X['log_wap_ratio_index_wap'] = np.log1p(_X['wap']/_X['mean_index_wap'])
        
        # Money flow index
        # _X = self.money_flow_index(_X, ['wap', 'mean_index_wap', 'mid_price'], 'bid_plus_ask_sizes')
        
        # Time feats
        _X["dow"] = _X["date_id"] % 5  # Day of the week
        _X["dom"] = _X["date_id"] % 22  # Day of the month
        _X["seconds"] = _X["seconds_in_bucket"] % 60  # Seconds
        _X["minute"] = _X["seconds_in_bucket"] // 60  # Minutes
        
        
        _X = self.fillna(_X)
        
        return _X
        
    def get_median_std_values(self, X):
        for k in self.stocks_info.keys():
            X[k] = (X['stock_id'].map(self.stocks_info[k].to_dict())).astype(np.float32)
        
        return X
    
    def get_time_statistics(self, X):
        gb = X.groupby(['date_id', 'seconds_in_bucket'])

        t = {
            'mean_index_bid_price':gb['bid_price'].transform('mean'),
            'mean_index_ask_price':gb['ask_price'].transform('mean'),
            'mean_index_bid_size':gb['bid_size'].transform('mean'),
            'mean_index_ask_size':gb['ask_size'].transform('mean'),
            
            'mean_index_wap_raw':gb['wap'].transform('mean'),
            'std_index_wap_raw':gb['wap'].transform('std'),
            'skew_index_wap_raw':gb['wap'].transform('skew'),
            # 'kurt_index_wap_raw':gb['wap'].transform(pd.DataFrame.kurt),
            
            'std_index_bid_size':gb['bid_size'].transform('std'),
            'std_index_ask_size':gb['ask_size'].transform('std'),
            
            'min_index_bid_size':gb['bid_size'].transform('min'),
            'min_index_ask_size':gb['ask_size'].transform('min'),
            
            'max_index_bid_size':gb['bid_size'].transform('max'),
            'max_index_ask_size':gb['ask_size'].transform('max'),
            
            'skew_index_bid_size':gb['bid_size'].transform('skew'),
            'skew_index_ask_size':gb['ask_size'].transform('skew'),
            
            'norm_bid_size':gb['bid_size'].transform(np.linalg.norm),
            'norm_ask_size':gb['ask_size'].transform(np.linalg.norm),
            'norm_sum_size':gb['bid_plus_ask_sizes'].transform(np.linalg.norm),
            'norm_size_spread':gb['size_spread'].transform(np.linalg.norm),
        }

        return X.assign(**t)
    
    def get_stock_weights(self, X):
        X['matched_volume'] = X['matched_size'] * X['wap']

        sample = X.groupby(['date_id', 'seconds_in_bucket'], group_keys=False, sort=False)

        new_cols = {
            'weights':X['matched_volume']/sample['matched_volume'].transform('sum'),
        }

        return X.assign(**new_cols)

        
    def pair_imbalance(self, X, features):        
        # for c in combinations(features, 2):
        for c in features:
            X[f'{c[0]}_minus_{c[1]}'] = (X[f'{c[0]}'] - X[f'{c[1]}']).astype(np.float32)
            X[f'{c[0]}_{c[1]}_imb'] = (X[f'{c[0]}_minus_{c[1]}']/(X[c[0]]+X[c[1]])).astype(np.float32)
            
        return X
    
    def triple_imbalance(self, X, features):        
        # for c in combinations(features, 3):
        for c in features:
            max_ = X[list(c)].max(axis=1)
            min_ = X[list(c)].min(axis=1)
            mid_ = X[list(c)].sum(axis=1)-min_-max_

            X[f'{c[0]}_{c[1]}_{c[2]}_imb2'] = (max_-mid_)/(mid_-min_)
            
        return X
    
    def past_day_target(self, X):
        if 'past_day_target' not in X.columns:
            sample = X.groupby(['stock_id', 'seconds_in_bucket'], group_keys=False, sort=False)['target']

            X['past_day_target'] = sample.shift()
            X = self.change_features(X, ['past_day_target'], windows=[-12, -6, -2, -1, 1, 6])

        return X
    
    def rolling_features(self, X, features, windows=[5, 10, 20, 55]):
        sample = X.groupby(['stock_id', 'date_id'], group_keys=False, sort=False)[features]

        for window in windows:
            sma_new_feats = [f'rolling_{window}_{feature}' for feature in features]
            historical_volatility = [f'historical_volatility_{window}_{feature}' for feature in features]
            bbw_new_feats = [f'bbw_{window}_{feature}' for feature in features]
            coeficient_variance = [f'coeficient_variance_{window}_{feature}' for feature in features]
            
            base_sma_diff = [f'feature_diff_rolling_{window}_{feature}' for feature in features]
            base_sma_ratio = [f'feature_ratio_rolling_{window}_{feature}' for feature in features]
            
            max_ = sample.rolling(window, min_periods=1).max(engine="numba")
            min_ = sample.rolling(window, min_periods=1).min(engine="numba")
            # sma = sample.rolling(window, min_periods=1).mean(engine="numba")
            # vol = sample.rolling(window, min_periods=1).std(engine="numba") * np.sqrt(window)
            
            sma = sample.transform(lambda x: x.ewm(span=window).mean())
            vol = sample.transform(lambda x: x.ewm(span=window).std())
            
            X[sma_new_feats] = (sma).astype(np.float32)
            X[bbw_new_feats] = ((max_ - min_)/sma).astype(np.float32)
            X[historical_volatility] = (vol).astype(np.float32)
            X[coeficient_variance] = (vol/sma).astype(np.float32)
            
            X[base_sma_diff] = X[features].values - X[sma_new_feats].values
            X[base_sma_ratio] = X[features] / X[base_sma_diff].values
            
        # Difference between the multiple windows
        # for feature in features:
        #     feats = [f'rolling_{window}_{feature}' for window in windows[::-1]]
            
        #     X = self.pair_imbalance(X, combinations(feats, 2))
            
        return X
    
    def MACD(self, X, features, short_window, long_window, signal_window):
        macd_features = []
        signal_features = []
        
        for feature in features:
            macd_features.append(f'MACD_{feature}_{short_window}_{long_window}')
            signal_features.append(f'Signal_{signal_window}_{macd_features[-1]}')
            X[f'{macd_features[-1]}'] = X[f'rolling_{short_window}_{feature}'] - X[f'rolling_{long_window}_{feature}']
        
        sample = X.groupby(['stock_id', 'date_id'], group_keys=False, sort=False)[macd_features]
        X[signal_features] = sample.rolling(signal_window, min_periods=1).mean(engine="numba")
        
        for macd_feature, signal_feature in zip(macd_features, signal_features):
            X = self.pair_imbalance(X, [[macd_feature, signal_feature]])
            X = self.change_features(X, [f'{macd_feature}_minus_{signal_feature}'], [1, 6])
            
        return X
    
    def RSI(self, X, features, windows):
        _X = X[features + ['stock_id', 'date_id']].copy()
        change = _X.groupby(['stock_id', 'date_id'], group_keys=False, sort=False)[features].diff()
        
        for feature in features:
            change_feature = change[feature]

            __X = _X[[feature] + ['stock_id', 'date_id']].copy()
            
            __X['change_up'] = _X[feature].copy()
            __X['change_down'] = _X[feature].copy()
            
            __X.loc[change_feature<0, 'change_up'] = 0
            __X.loc[change_feature>0, 'change_down'] = 0
            __X.loc[:, 'change_down'] *= -1
            
            gb = __X.groupby(['stock_id', 'date_id'], group_keys=False, sort=False)[['change_up', 'change_down']]
            
            for window in windows:
                mean = gb.rolling(window, min_periods=1).mean(engine="numba")
                
                rs = mean['change_up']/mean['change_down']
                rsi = 100 - (100 / (1+rs))
                
                X[f'RSI_{window}_{feature}'] = rsi
                
                # gb_rsi = X.groupby(['stock_id', 'date_id'], group_keys=False, sort=False)[f'RSI_{window}_{feature}']
                
                # min_ = gb_rsi.rolling(12, min_periods=1).min(engine="numba")
                # max_ = gb_rsi.rolling(12, min_periods=1).max(engine="numba")
                
                # X[f'Stochastic_RSI_{window}_{feature}'] = (rsi - min_)/(max_-min_)
                
        return X
    
    def change_features(self, X, features, windows=[1, 2, 3, 10]):
        sample = X.groupby(['stock_id', 'date_id'], group_keys=False, sort=False)[features]
        
        for window in windows:
            shift_feats = [f"{feature}_shift_{window}" for feature in features]
            pct_change = [f"{feature}_pct_change_{window}" for feature in features]
            
            X[shift_feats] = sample.shift(window)
            X[pct_change] = sample.pct_change(window)
            
        return X
    
    def diff_features(self, X, features, windows=[1, 2, 3, 10]):
        sample = X.groupby(['stock_id', 'date_id'], group_keys=False, sort=False)[features]
        
        for window in windows:
            diff_feats = [f"{feature}_diff_{window}" for feature in features]
            
            X[diff_feats] = sample.shift(window)
            
        return X
    
    def corr(self, X, features, windows):
        for feat_1, feat_2 in features:
            gb = X.groupby(['stock_id', 'date_id'], group_keys=False, sort=False)[[feat_1, feat_2]]

            for window in windows:
                X[f'corr_{window}_{feat_1}_{feat_2}'] = gb.rolling(window, min_periods=1).corr().loc[:,:,:, feat_1][feat_2].reset_index(level=[0,1], drop=True)

        return X
    
    def get_day_high_low(self, X, features):
        sample = X.groupby(['stock_id', 'date_id'], group_keys=False, sort=False)[features]
        
        max_feats = [f'{feature}_high' for feature in features]
        min_feats = [f'{feature}_low' for feature in features]
        
        max_ = sample.rolling(55, min_periods=1).max(engine="numba")
        min_ = sample.rolling(55, min_periods=1).min(engine="numba")
        
        X[max_feats] = max_
        X[min_feats] = min_
        
        return X
    
    def money_flow_index(self, X, features, volume):
        X = self.get_day_high_low(X, features)
        money_flow_features = []
        
        for feature in features:
            X[f'typical_price_{feature}'] = X[[f'{feature}_low', f'{feature}', f'{feature}_high']].mean(axis=1)
            X[f'money_flow_{feature}'] = X[f'typical_price_{feature}'] * X[volume]
            
            money_flow_features.append(f'money_flow_{feature}')
        
        X = self.RSI(X, money_flow_features, windows=[55])
        
        return X
    
    def fillna(self, X):
        X = X.replace([np.inf, -np.inf], 0)
        
        return X.fillna(0)
        # return X.groupby(['stock_id', 'date_id'], group_keys=False, sort=False).fillna(method='ffill').fillna(0)