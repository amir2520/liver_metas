from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from collections import Counter


@dataclass
class GeneMutProcessConfig:
	_target_: str = 'metastatic.preprocess.gene_mutation_transformer.GeneMutProcess'
	_partial_: bool = True
	gene_counter: Counter = MISSING
	include_rare: bool = True
	rare_threshold: int = 20
	gene_col: str = 'gene_mut'


def setup_config():
	cs = ConfigStore.instance()
	cs.store(
		name='gene_mutation_precessor_schema',
		node=GeneMutProcessConfig,
		group='preprocessing'
	)