import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
# from metastatic.config_schemas.pipeline_schemas.pipeline_config_schema import setup_config
from metastatic.config_schemas.config_schema import setup_config
from metastatic.utils.utils import space_tokenizer

# import pandas as pd
# import numpy as np
# import pickle
# from sklearn.pipeline import Pipeline


# with open('metastatic/datasets/gene_mut_counter.pkl', 'rb') as file:
# 	gene_counter = pickle.load(file)

# X_train = pd.read_csv('metastatic/datasets/X_train_liver_all.csv')
# y_train = pd.read_csv('metastatic/datasets/y_train_liver_all.csv').values.ravel()

setup_config()

@hydra.main(config_path = 'configs', config_name = 'config', version_base = None)
def main(config: DictConfig):
	
	print(OmegaConf.to_yaml(config))
	# tokenizer = space_tokenizer
	# model_pipeline = instantiate(config.pipeline, gene_counter=gene_counter, tokenizer=tokenizer)
	# model_pipeline.fit(X_train, y_train)
	# print(f'predict value: {model_pipeline.predict(X_train.iloc[[1], :])}')
	# print(f'predict_proba value: {model_pipeline.predict_proba(X_train.iloc[[1], :])}')
	# for step in model_pipeline.pipeline.steps:
	# 	print(step)
	# 	print()
	# 	print()
	# print(np.unique(y_train))
	# pipeline.fit(X_train, y_train)
	# partial_gene_process = instantiate(config.gene_mutation_preprocess)
	# gene_process = partial_gene_process(gene_counter = gene_counter)

	# model_pipeline = Pipeline([
	# 	('gene_process', gene_process)
	# ])
	# X_processed = model_pipeline.fit_transform(X_train)
	# print(X_processed)


if __name__ == "__main__":
	main()