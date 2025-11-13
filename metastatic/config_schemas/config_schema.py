from dataclasses import dataclass, field
from omegaconf import MISSING
from collections import Counter
from typing import Callable, Optional
from metastatic.config_schemas import dimension_reduction_schemas
from metastatic.config_schemas import gene_mutation_transformer_schema
from metastatic.config_schemas import model_schemas
from metastatic.config_schemas import vectorizer_schemas
from hydra.core.config_store import ConfigStore


@dataclass
class PipelineConfig:
	dim_reduction_layer: Optional[dimension_reduction_schemas.DimensionalityReductionConfig] = dimension_reduction_schemas.NMFConfig()
	partial_vectorizer_layer: Optional[vectorizer_schemas.VectorizerConfig] = vectorizer_schemas.PartialTfidfVectorizerConfig()
	partial_gene_mutation_preprocess_layer: gene_mutation_transformer_schema.PartialGeneMutProcessConfig = gene_mutation_transformer_schema.PartialGeneMutProcessConfig()
	gene_counter: Counter = MISSING
	tokenizer: Callable[[str], list[str]] = MISSING


@dataclass
class SingleModelPipelineConfig(PipelineConfig):
	_target_: str = MISSING
	model_layer: model_schemas.ModelConfig = model_schemas.LogisticRegressionConfig()


@dataclass
class VotingEnsemblePipelineConfig(PipelineConfig):
	_target_: str = MISSING
	model_layers: list[model_schemas.ModelConfig] = MISSING #field(default_factory=lambda: [])


def setup_config():
	dimension_reduction_schemas.setup_config()
	gene_mutation_transformer_schema.setup_config()
	model_schemas.setup_config()

	cs = ConfigStore.instance()
	cs.store(name = 'config_schema', node = PipelineConfig)