from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class ModelConfig:
	_target_: str = MISSING


@dataclass
class LogisticRegressionConfig(ModelConfig):
	_target_: str = 'sklearn.linear_model.LogisticRegression'
	C: float = 10.0
	class_weight: dict = field(default_factory=lambda: {0:1, 1:1})


@dataclass
class RandomForestClassifierConfig(ModelConfig):
	_target_: str = 'sklearn.ensemble.RandomForestClassifier'
	max_depth: int = 5
	class_weight: dict = field(default_factory=lambda: {0: 1, 1: 1})


@dataclass
class LinearSVCConfig(ModelConfig):
	_target_: str = 'sklearn.svm.LinearSVC'
	max_iter: int = 1000
	class_weight: dict = field(default_factory=lambda: {0: 1, 1: 1})



