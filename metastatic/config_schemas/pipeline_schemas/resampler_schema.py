from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from typing import Optional


@dataclass
class SMOTEConfig:
	_target_: str = 'imblearn.over_sampling.SMOTE'
	sampling_strategy: str = 'auto'
	k_neighbors: int = 5
	random_state: Optional[int] = None



def setup_config():
	cs = ConfigStore.instance()
	cs.store(
		name="SMOTE_schema",
		node=SMOTEConfig,
		group='imbalance'
	)