from typing import Protocol, Union, Optional, Callable
from collections import Counter
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from imblearn.base import BaseSampler



class PartialGeneMutProcessType(Protocol):
	def __call__(self, gene_counter: Counter) -> GeneMutProcess:
		...

class PartialTfidfVectorizerType(Protocol):
	def __call__(self, tokenizer: Callable[[str], list[str]]) -> TfidfVectorizer:
		...


class BaseSklearnPipeline(ABC):
	def __init__(
		self,
		gene_counter: Counter,
		tokenizer: Callable[[str], list[str]],
		partial_gene_mutation_preprocess_layer: PartialGeneMutProcessType,
		partial_vectorizer_layer: Optional[PartialTfidfVectorizerType] = None,
		dim_reduction_layer: Optional[BaseEstimator] = None,
		resampler_layer: Optional[BaseSampler] = None,
		scaler_layer: Optional[BaseEstimator] = None
	) -> None:

		gene_mutation_layer = partial_gene_mutation_preprocess_layer(gene_counter=gene_counter)

		self.gene_mutation_layer = gene_mutation_layer
		self.tfidf_vectorizer = partial_vectorizer_layer(tokenizer=tokenizer) if partial_vectorizer_layer else None
		self.dim_reduction_layer = dim_reduction_layer
		self.resampler_layer = resampler_layer
		self.scaler_layer = scaler_layer

	@abstractmethod
	def build_pipeline(self) -> Pipeline:
		...

	@abstractmethod
	def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseSklearnPipeline":
		...

	@abstractmethod
	def predict(self, X: pd.DataFrame) -> np.ndarray:
		...

	@abstractmethod
	def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
		...








