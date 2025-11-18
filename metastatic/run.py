import hydra
from hydra.utils import instantiate, get_method
from omegaconf import DictConfig, OmegaConf
from metastatic.config_schemas.config_schema import setup_config
from metastatic.utils.utils import activate_mlflow
import pandas as pd
from metastatic.utils.utils import log_training_hparams_single_model
import mlflow



# X_train = pd.read_csv('metastatic/datasets/X_train_liver_all.csv')
# y_train = pd.read_csv('metastatic/datasets/y_train_liver_all.csv').values.ravel()

setup_config()


@hydra.main(config_path = 'configs', config_name = 'config', version_base = None)
def main(config: DictConfig):
	
	OmegaConf.register_new_resolver("get_method", get_method, replace=True)
	# print(OmegaConf.to_yaml(config, resolve=True))
	
	X_train = pd.read_csv(config.single_model.infrastructure.data.X_train_path)
	y_train = pd.read_csv(config.single_model.infrastructure.data.y_train_path).values.ravel()

	with activate_mlflow(experiment_name=config.single_model.infrastructure.mlflow.experiment_name) as _:
		model_pipeline = instantiate(config.single_model.pipeline)
		scores = model_pipeline.cross_validation(X_train, y_train, scoring='f1')

		log_training_hparams_single_model(config=config)

		mlflow.log_metric('crossval_f1_mean': scores.mean())




if __name__ == "__main__":
	main()