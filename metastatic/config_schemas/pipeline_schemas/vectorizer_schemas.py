from dataclasses import dataclass
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
from typing import Callable, Any
from collections.abc import Callable


@dataclass
class VectorizerConfig:
	_target_: str = MISSING


@dataclass
class PartialTfidfVectorizerConfig(VectorizerConfig):
	_target_: str = 'sklearn.feature_extraction.text.TfidfVectorizer'
	_partial_: bool = True
	# tokenizer: Callable[[str], list[str]] = MISSING
	tokenizer: Any = MISSING



def setup_config():
	cs = ConfigStore.instance()
	cs.store(
		name="tfidf_vectorizer_schema",
		node=PartialTfidfVectorizerConfig,
		group="vectorizers"
	)