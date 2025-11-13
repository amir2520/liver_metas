from dataclasses import dataclass
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore


@dataclass
class DimensionalityReductionConfig:
	_target_: str = MISSING


@dataclass
class NMFConfig(DimensionalityReductionConfig):
	_target_: str = "sklearn.decomposition.NMF"
	n_components: int = 36
	max_iter: int = 1000


@dataclass
class PCAConfig(DimensionalityReductionConfig):
	_target_: str = 'sklearn.decomposition.PCA'
	n_components: int = 50


def setup_config():
	cs = ConfigStore.instance()
	cs.store(
		name="nmf_schema",
		node=NMFConfig,
		group="dim_reduction"
	)

	cs.store(
		name='pca_schema',
		node=PCAConfig,
		group="dim_reduction"
	)