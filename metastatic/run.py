import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from metastatic.config_schemas.config_schema import setup_config
from metastatic.utils.utils import space_tokenizer
from metastatic.utils.utils import activate_mlflow
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline


with open('metastatic/datasets/gene_mut_counter.pkl', 'rb') as file:
	gene_counter = pickle.load(file)

X_train = pd.read_csv('metastatic/datasets/X_train_liver_all.csv')
y_train = pd.read_csv('metastatic/datasets/y_train_liver_all.csv').values.ravel()

setup_config()

@hydra.main(config_path = 'configs', config_name = 'config', version_base = None)
def main(config: DictConfig):
	
	# print(OmegaConf.to_yaml(config, resolve=True))
	# tokenizer = space_tokenizer

	# with activate_mlflow(experiment_name=config.single_model.infrastructure.mlflow.experiment_name) as _:
	# 	model_pipeline = instantiate(config.single_model.pipeline, gene_counter=gene_counter, tokenizer=tokenizer)
	# 	# mlflow.log_param('C', config.single_model.pipeline.model_layer.C)
	# 	mlflow.sklearn.autolog()

	# 	model_pipeline.fit(X_train, y_train)
		# print(f'predict value: {model_pipeline.predict(X_train.iloc[[1], :])}')
	# print(f'predict_proba value: {model_pipeline.predict_proba(X_train.iloc[[1], :])}')
	# for step in model_pipeline.pipeline.steps:
	# 	print(step)
	


if __name__ == "__main__":
	main()