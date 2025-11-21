import hydra
from hydra.utils import instantiate, get_method
from omegaconf import DictConfig, OmegaConf
from metastatic.config_schemas.config_schema import setup_config
from metastatic.utils.utils import activate_mlflow, log_model
import pandas as pd
from metastatic.utils.utils import log_training_hparams_single_model
import mlflow


 


setup_config()


@hydra.main(config_path = 'configs', config_name = 'config', version_base = None)
def main(config: DictConfig):

	OmegaConf.register_new_resolver("get_method", get_method, replace=True)

	# print(OmegaConf.to_yaml(config, resolve=True))
	
	X_train = pd.read_csv(config.single_model.infrastructure.data.X_train_path)
	y_train = pd.read_csv(config.single_model.infrastructure.data.y_train_path).values.ravel()

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
		print("DEBUG run.py - best_run_data:", model_selector.best_run_data)
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