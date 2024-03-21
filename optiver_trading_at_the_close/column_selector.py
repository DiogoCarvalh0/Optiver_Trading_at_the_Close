from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_drop=None, cols_to_keep=None) -> None:
        super().__init__()
        
        self.cols_to_drop = cols_to_drop
        self.cols_to_keep = cols_to_keep
        
        if self.cols_to_drop and self.cols_to_keep:
            raise ValueError('cols_to_keep and cols_to_drop can not be used at the same time.')
        
    def fit(self, X, y=None):
        if self.cols_to_drop:
            self._cols_to_keep = [col for col in X.columns if col not in self.cols_to_drop]
        elif self.cols_to_keep:
            self._cols_to_keep = self.cols_to_keep
        else:
            self._cols_to_keep = X.columns
        
        self.is_fit_ = True
        
        return self
        
    def transform(self, X):
        check_is_fitted(self, 'is_fit_')
        
        return X[self._cols_to_keep]
