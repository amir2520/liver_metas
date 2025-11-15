from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from metastatic.config_schemas.infrastructure_schemas import mlflow_data_schema
from metastatic.config_schemas.pipeline_schemas import pipeline_config_schema


@dataclass
class BaseConfig:
	infrastructure: mlflow_data_schema.InfrastructureConfig = mlflow_data_schema.InfrastructureConfig()


@dataclass
class SingleModelConfig(BaseConfig):
	pipeline: pipeline_config_schema.SingleModelPipelineConfig = pipeline_config_schema.SingleModelPipelineConfig()



def setup_config():
	mlflow_data_schema.setup_config()
	pipeline_config_schema.setup_config()

	cs = ConfigStore.instance()
	cs.store(
		name='single_model_config_schema',
		node=SingleModelConfig,
		group='single_model'
	)

