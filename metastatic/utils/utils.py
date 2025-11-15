from typing import Optional, Union, Any, Iterable
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from contextlib import contextmanager
from mlflow.tracking.fluent import ActiveRun
import mlflow


def space_tokenizer(s: str) -> list[str]:
	return s.split()


class SparseToArray(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        pass
    
    def fit(self, X: Union[np.ndarray, sp.spmatrix], y: Optional[Union[np.ndarray, pd.Series]] = None) -> "SparseToArray":
        return self
    
    def transform(self, X: Union[np.ndarray, sp.spmatrix]) -> np.ndarray:
        return X.toarray() if sp.issparse(X) else X



@contextmanager
def activate_mlflow(
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
    run_id: Optional[str] = None
) -> Iterable[mlflow.ActiveRun]:
    set_experiment(experiment_name)

    run: ActiveRun
    with mlflow.start_run(run_name=run_name, run_id=run_id) as run:
        yield run


def set_experiment(experiment_name: Optional[str] = None) -> None:
    if experiment_name is None:
        experiment_name = 'Default'

    try:
        mlflow.create_experiment(experiment_name)
    except mlflow.exceptions.RestException:
        pass

    mlflow.set_experiment(experiment_name)




