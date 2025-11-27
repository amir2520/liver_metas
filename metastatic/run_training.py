import pandas as pd
import dvc.api
import hydra
import mlflow
from functools import partial
from hydra.utils import instantiate, get_method
from omegaconf import DictConfig, OmegaConf
from metastatic.config_schemas.config_schema import setup_config
from metastatic.utils.utils import activate_mlflow, log_model
from metastatic.utils.utils import log_training_hparams_single_model
from metastatic.utils.gcp_utils import access_secret_version 
from metastatic.utils.data_utils import get_data_url_with_version

import os
os.environ['HYDRA_FULL_ERROR'] = '1'
 
setup_config()


@hydra.main(config_path = 'configs', config_name = 'config', version_base = None)
def main(config: DictConfig):

	OmegaConf.register_new_resolver("get_method", get_method, replace=True)

	# print(OmegaConf.to_yaml(config, resolve=True))
	
	# X_train = pd.read_csv(config.single_model.infrastructure.data.X_train_path)
	# y_train = pd.read_csv(config.single_model.infrastructure.data.y_train_path).values.ravel()

	github_access_token = access_secret_version(project_id=config.single_model.infrastructure.data.gcp_project_id, 
												secret_id=config.single_model.infrastructure.data.github_access_token_secret_id)

	data_url = partial(get_data_url_with_version,
				  		  version=config.single_model.infrastructure.data.version,
						  dvc_remote_repo=config.single_model.infrastructure.data.dvc_remote_repo,
						  dvc_data_folder=config.single_model.infrastructure.data.dvc_data_folder,
						  github_user_name=config.single_model.infrastructure.data.github_user_name,
						  github_access_token=github_access_token
						  )

	X_train_url = data_url(dataset_name=config.single_model.infrastructure.data.X_train_path)
	X_test_url = data_url(dataset_name=config.single_model.infrastructure.data.X_test_path)
	y_train_url = data_url(dataset_name=config.single_model.infrastructure.data.y_train_path)
	y_test_url = data_url(dataset_name=config.single_model.infrastructure.data.y_test_path)

	X_train = pd.read_csv(X_train_url)
	y_train = pd.read_csv(y_train_url).values.ravel()

	with activate_mlflow(experiment_name=config.single_model.infrastructure.mlflow.experiment_name) as run:

		run_id = run.info.run_id
		config.single_model.model_selector.mlflow_run_id = run_id
		config.single_model.infrastructure.mlflow.run_id = run_id
		
		model_pipeline = instantiate(config.single_model.pipeline)
		scores = model_pipeline.cross_validation(X_train, y_train, scoring='f1')

		log_training_hparams_single_model(config=config)
		mlflow.log_metric('crossval_f1_mean', scores['mean'])



	model_selector = instantiate(config.single_model.model_selector)
	assert config.single_model.registered_model_name is not None
	if model_selector is not None:
		if model_selector.is_selected():
			# Fit the model only when it's the best
			model_pipeline.fit(X_train, y_train)
			log_model(
				config.single_model.infrastructure.mlflow, 
				model_selector.get_new_best_run_tag(), 
				config.single_model.registered_model_name,
				model_pipeline  
			)



if __name__ == "__main__":
	main()