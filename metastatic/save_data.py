import hydra
from hydra.utils import instantiate, get_method
from omegaconf import DictConfig, OmegaConf
from metastatic.config_schemas.config_schema import setup_config
from metastatic.utils.utils import activate_mlflow, log_model
import pandas as pd
from metastatic.utils.utils import log_training_hparams_single_model
import mlflow
from metastatic.utils.gcp_utils import access_secret_version 
from metastatic.utils.data_utils import get_raw_data_with_version, get_data_url_with_version
import dvc.api
from functools import partial




setup_config()


@hydra.main(config_path = 'configs', config_name = 'config', version_base = None)
def main(config: DictConfig):

	OmegaConf.register_new_resolver("get_method", get_method, replace=True)

	# print(OmegaConf.to_yaml(config, resolve=True))

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



	df = pd.read_csv(X_train_url)
	print(df.head())

	# version = 'v3'
	# data_local_save_dir = './data/raw'
	# dvc_remote_repo = 'https://github.com/amir2520/metastasis-data-versioning.git'
	# dvc_data_folder = 'data/raw'
	# github_user_name = 'amir2520'
	# github_access_token = access_secret_version(project_id='end-to-end-ml-course-466603',
	# 											secret_id='metastasis_data_github_access_token')

	
	
	# without_https = dvc_remote_repo.replace('https://', '')
	# dvc_remote_repo = f'https://{github_user_name}:{github_access_token}@{without_https}'
	# X_train_url = dvc.api.get_url('data/raw/X_train_liver_all.csv', repo=dvc_remote_repo, rev=version)
	# df = pd.read_csv(X_train_url)
	# print(df.head())



if __name__ == "__main__":
	main()