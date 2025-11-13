from dataclasses import dataclass
from metastatic.config_schemas import dimension_reduction_schemas
from metastatic.config_schemas import gene_mutation_transformer_schema
from hydra.core.config_store import ConfigStore


@dataclass
class Config:
	dim_reduction: dimension_reduction_schemas.DimensionalityReductionConfig = dimension_reduction_schemas.NMFConfig()
	gene_mutation_preprocess: gene_mutation_transformer_schema.GeneMutProcessConfig = gene_mutation_transformer_schema.GeneMutProcessConfig()


def setup_config():
	dimension_reduction_schemas.setup_config()
	gene_mutation_transformer_schema.setup_config()

	cs = ConfigStore.instance()
	cs.store(name = 'config_schema', node = Config)