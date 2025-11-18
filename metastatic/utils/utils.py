from typing import Optional, Union, Any, Iterable, TYPE_CHECKING, Generator
import dataclasses
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from contextlib import contextmanager
from mlflow.tracking.fluent import ActiveRun
import mlflow
from omegaconf import DictConfig, OmegaConf, MISSING
if TYPE_CHECKING:
    from metastatic.config_schemas.config_schema import SingleModelConfig



def space_tokenizer(s: str) -> list[str]:
	return s.split()


class LoggableParamsMixin:
    def loggable_params(self) -> list[str]:
        return []


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


def log_training_hparams_single_model(config: "SingleModelConfig") -> None:
    structured_config = OmegaConf.to_object(config.single_model)
    logged_nodes = set()
    def loggable_params(node: Any, path: list[str]) -> Generator[tuple[str, Any], None, None]:
        current_path = ".".join(path) if path else "root"       
        if isinstance(node, LoggableParamsMixin) and id(node) not in logged_nodes:
            for param_name in node.loggable_params():
                param_value = getattr(node, param_name)
                full_key = ".".join(path + [param_name])
                yield full_key, param_value
            logged_nodes.add(id(node))
        
        children = None
        if isinstance(node, (dict, DictConfig)):
            try:
                children = node.items()
            except Exception as e:
                return
        elif dataclasses.is_dataclass(node):
            children = ((f.name, getattr(node, f.name)) for f in dataclasses.fields(node))
        if children is None:
            return
        
        for key, val in children:
            if isinstance(val, type(MISSING)):
                continue
            try:
                for item in loggable_params(val, path + [key]):
                    yield item
            except Exception as e:
                continue
    
    params = dict(loggable_params(structured_config, []))
    mlflow.log_params(params)



