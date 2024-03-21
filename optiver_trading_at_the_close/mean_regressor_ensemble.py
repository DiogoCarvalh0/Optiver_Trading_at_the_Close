import os
from tqdm import tqdm
import copy

import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import TimeSeriesSplit


class MeanRegressorEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, estimators, weights=None) -> None:
        super().__init__()
        
        if weights != None:
            assert (len(estimators) == len(weights))
        
        self.estimators = estimators
        self.weights = np.ones(len(estimators))/len(estimators) if weights==None else weights
        
    def fit(self, X, y, fit_estimators=True, nr_cv_folds=None, **fit_parameters):  
        if fit_estimators:
            if nr_cv_folds != None:
                estimators = [self.__fit_cv(estimator, X, y, nr_cv_folds, **fit_parameters) for estimator in self.estimators]
                self.estimators = [item for sublist in estimators for item in sublist]
                
                # TODO: accept if weights were not uniform
                self.weights = np.ones(len(self.estimators))/len(self.estimators)
                
            else:
                self.estimators = [estimator.fit(X, y, **fit_parameters) for estimator in tqdm(self.estimators)]
        
        self.is_fit_ = True
        
        return self
    
    def __fit_cv(self, estimator, X, y, nr_folds, **fit_parameters):
        estimators = []
        
        for train_index, test_index in tqdm(TimeSeriesSplit(n_splits=nr_folds-1).split(X), total=nr_folds-1):
            model = copy.deepcopy(estimator)
            model.fit(
                X.iloc[train_index],
                y.iloc[train_index],
                eval_set=[(X.iloc[test_index], y.iloc[test_index])],
                **fit_parameters
            )
            estimators.append(model)
        
        # Train a model with all the data
        model = copy.deepcopy(estimator)
        try: # use averange number of trees if possible else keep the baseline
            average_best_iteration = int(np.mean([model.best_iteration_ for model in estimators]))
            model.set_params(**{'n_estimators':average_best_iteration})
        except:
            pass
        model.fit(X.values, y.values)
        estimators.append(model)
        
        return estimators
        
    def predict(self, X, index=None):
        check_is_fitted(self, 'is_fit_')
        
        index_to_predict = X.index if index is None else index
        
        _X = X.loc[index_to_predict]
        
        preds = [estimator.predict(_X) for estimator in self.estimators]
        
        return np.average(preds, axis=0, weights=self.weights)
        