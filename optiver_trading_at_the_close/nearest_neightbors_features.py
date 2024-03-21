import os

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

class NearestNeighborsFeatures(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        get_target=True,
        features_to_use_for_distance_computation=None,
        features_get=None,
        metrics=['l1', 'l2'],
        n_neighbors=[2, 3, 5, 10, 20, 40],
        exclude_self=True,
        n_jobs=1
    ) -> None:
        super().__init__()
        
        if (not get_target) and (not features_get):
            raise ValueError("Need to have at least 1 feature to use to create new features")
        
        self.get_target = get_target
        self.features_to_use_for_distance_computation = features_to_use_for_distance_computation
        self.features_get = features_get.copy()
        self._features_get = features_get.copy() if features_get else []
        if self.get_target:
            self._features_get.append('target')
            
        self.metrics = metrics
        self.n_neighbors = n_neighbors
        self.max_neighbors = max(self.n_neighbors) + 1 if exclude_self else max(self.n_neighbors)
        self.exclude_self = exclude_self
        self.n_jobs = os.cpu_count() if n_jobs==-1 else n_jobs
        
        self._features_pivot = dict()
        
        
    def fit(self, X, y):
        self.features_to_use_for_distance_computation = self.features_to_use_for_distance_computation if self.features_to_use_for_distance_computation else X.columns

        pivot = X.pivot(index='time_id', columns='stock_id', values=self.features_to_use_for_distance_computation)
        pivot = pivot.fillna(pivot.mean())

        self.scaler = MinMaxScaler()
        self.scaler.fit(pivot)
        pivot = pd.DataFrame(self.scaler.transform(pivot))

        self.nearest_neighbors_estimators = [NearestNeighbors(n_neighbors=self.max_neighbors, metric=metric, n_jobs=self.n_jobs).fit(pivot) for metric in self.metrics]

        if self.features_get:            
            for feature in self.features_get:
                feature_pivot = X.pivot(index='time_id', columns='stock_id', values=feature)
                feature_pivot = feature_pivot.fillna(feature_pivot.mean())
                self._features_pivot[feature] = feature_pivot
                
        if self.get_target:
            _X = X[['time_id', 'stock_id']].copy()
            _X['target'] = y 
    
            _target_pivot = _X.pivot(index='time_id', columns='stock_id', values='target')
            _target_pivot = _target_pivot.fillna(_target_pivot.mean())
            self._features_pivot['target'] = _target_pivot
            
        self.is_fit_ = True
        
        return self
        
    def transform(self, X):
        check_is_fitted(self, 'is_fit_')
        
        pivot = X.pivot(index='time_id', columns='stock_id', values=self.features_to_use_for_distance_computation)
        pivot = pivot.fillna(pivot.mean())
        pivot = pd.DataFrame(self.scaler.transform(pivot), index=pivot.index)
            
        estimator_neighbors = [self.__get_neighbors(pivot, estimator) for estimator in self.nearest_neighbors_estimators]
            
        for i, neighbors in enumerate(estimator_neighbors):
            for feature in self._features_get:
                X = self.__add_features(
                    X=X,
                    pivot=pivot,
                    target_feature_pivot=self._features_pivot[feature],
                    neighbors=neighbors,
                    nr_neighbors_for_metrics=self.n_neighbors,
                    feature_name=f'{feature}_{self.metrics[i]}'
                )
                
        return X
    
    def __get_neighbors(self, X, estimator):
        dist, neighbors = estimator.kneighbors(X, return_distance=True)
        
        if self.exclude_self:
            if (dist == 0).sum().sum() > 0:
                neighbors = neighbors[:, 1:]
                
        return neighbors
    
    def __add_features(
        self,
        X:pd.DataFrame,
        pivot:pd.DataFrame,
        target_feature_pivot:pd.DataFrame,
        neighbors:list[list[int]],
        nr_neighbors_for_metrics:list[int],
        feature_name:str
    ):
        values = target_feature_pivot.values[neighbors]
        
        agg_values = {
            'stock_id':list(target_feature_pivot.columns) * len(pivot.index),
            'time_id':pivot.index.repeat(len(target_feature_pivot.columns))
        }

        for n in nr_neighbors_for_metrics:
            values_up_to_n = values[:, :n, :]
                     
            for agg in [np.mean, np.median, np.min, np.max, np.std]:
                agg_values[f'{feature_name}_{agg.__name__}_{n}_neighbors'] = agg(values_up_to_n, axis=1).flatten()
            
        agg_values = pd.DataFrame(agg_values)
        
        X = X.merge(agg_values, on=['stock_id', 'time_id'])
                
        return X 
