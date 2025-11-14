from dataclasses import dataclass
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore


@dataclass
class ScalerConfig:
	_target_: str = MISSING


@dataclass
class StandardScalerConfig(ScalerConfig):
	_target_: str = 'sklearn.preprocessing.StandardScaler'
	with_mean: bool = True
	with_std: bool = True


@dataclass
class MinMaxScalerConfig(ScalerConfig):
	_target_: str = 'sklearn.preprocessing.MinMaxScaler'
	feature_range: tuple[float, float] = (0, 1)



def setup_config():
	cs = ConfigStore.instance()
	cs.store(
		name='standard_scaler_schema',
		node=StandardScalerConfig,
		group='scalers'
	)

	cs.store(
		name='minmax_scaler_schema',
		node=MinMaxScalerConfig,
		group='scalers'
	)