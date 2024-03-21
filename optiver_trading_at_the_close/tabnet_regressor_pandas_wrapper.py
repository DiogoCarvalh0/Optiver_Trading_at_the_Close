import joblib

import pandas as pd

from pytorch_tabnet.tab_model import TabNetRegressor

class TabNetRegressorPandasWrapper(TabNetRegressor):
    def __init__(self, cat_variables:list, **tabnet_parameters):
        self.cat_variables = cat_variables
        self.tabnet_params = tabnet_parameters
        
    def fit(self, X:pd.DataFrame, y:pd.Series, load_model_path:str=None, **fit_params):
        if load_model_path:
            self.model = joblib.load(load_model_path)
            fit_params['warm_start'] = True
        else:
            self.model = self.__define_model(X)
        
        self.model.fit(X.values, y.values.reshape(-1, 1), **fit_params)
        
        return self
    
    def predict(self, X:pd.DataFrame):
        return self.model.predict(X.values).flatten()
        
    def __define_model(self, X:pd.DataFrame) -> TabNetRegressor:
        cat_idxs = [i for i, f in enumerate(X.columns) if f in self.cat_variables]
        cat_dims = [X[feature].nunique() for feature in self.cat_variables]
        
        self.tabnet_params['cat_idxs'] = cat_idxs
        self.tabnet_params['cat_dims'] = cat_dims
        
        return TabNetRegressor(**self.tabnet_params)