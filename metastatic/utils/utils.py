from typing import Optional, Union, Any
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


def space_tokenizer(s: str) -> list[str]:
	return s.split()


class SparseToArray(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        pass
    
    def fit(self, X: Union[np.ndarray, sp.spmatrix], y: Optional[Union[np.ndarray, pd.Series]] = None) -> "SparseToArray":
        return self
    
    def transform(self, X: Union[np.ndarray, sp.spmatrix]) -> np.ndarray:
        return X.toarray() if sp.issparse(X) else X