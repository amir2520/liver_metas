import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from metastatic.config_schemas.config_schema import setup_config


# import pandas as pd
# import pickle
# from sklearn.pipeline import Pipeline


# with open('metastatic/datasets/gene_mut_counter.pkl', 'rb') as file:
# 	gene_counter = pickle.load(file)

# X_train = pd.read_csv('metastatic/datasets/X_train_liver_all.csv')

setup_config()

@hydra.main(config_path = 'configs', config_name = 'config', version_base = None)
def main(config: DictConfig):
	print(OmegaConf.to_yaml(config))
	# partial_gene_process = instantiate(config.gene_mutation_preprocess)
	# gene_process = partial_gene_process(gene_counter = gene_counter)

	# model_pipeline = Pipeline([
	# 	('gene_process', gene_process)
	# ])
	# X_processed = model_pipeline.fit_transform(X_train)
	# print(X_processed)


if __name__ == "__main__":
	main()