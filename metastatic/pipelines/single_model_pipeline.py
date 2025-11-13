from metastatic.pipelines.base_pipeline import BaseSklearnPipeline
from sklearn.pipeline import Pipeline
from sklearn.base import ClassifierMixin
import pandas as pd
import numpy as np


class SingleModelPipeline(BaseSklearnPipeline):
	def __init__(self, model_layer: ClassifierMixin, **kwargs) -> None:
		super().__init__(**kwargs)
		self.model_layer = model_layer
		self.pipeline = self.build_pipeline()

	def build_pipeline(self) -> Pipeline:
		...

	def fit(self, X:pd.DataFrame, y: pd.Series) -> "SingleModelPipeline":
		...

	def predict(self, X: pd.DataFrame) -> np.ndarray:
		...

	def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
		...