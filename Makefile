 # Make all targets .PHONY
.PHONY: $(shell sed -n -e '/^$$/ { n ; /^[^ .\#][^ ]*:/ { s/:.*$$// ; p ; } ; }' $(MAKEFILE_LIST))

include .envs/.mlflow_dev
include .envs/.postgres
include .envs/.dataset_gcp_project


export

SHELL = /usr/bin/env bash
USER_NAME = $(shell whoami)
HOST_NAME = $(shell hostname)
USER_ID = $(shell id -u)

ifeq (, $(shell which docker-compose))
	DOCKER_COMPOSE_COMMAND = docker compose
else
	DOCKER_COMPOSE_COMMAND = docker-compose
endif

PROD_PROFILE=prod
PROD_CONTAINER_NAME=metas-model-prod-container
PROD_SERVICE_NAME=app-prod


PROFILE = ci
CONTAINER_NAME = metas-model-dev-container
SERVICE_NAME = app-ci

DOCKER_COMPOSE_RUN = $(DOCKER_COMPOSE_COMMAND) run --rm $(SERVICE_NAME)
DOCKER_COMPOSE_EXEC = $(DOCKER_COMPOSE_COMMAND) exec $(SERVICE_NAME)

PROD_DOCKER_COMPOSE_RUN = $(DOCKER_COMPOSE_COMMAND) run --rm $(PROD_SERVICE_NAME)
PROD_DOCKER_COMPOSE_EXEC = $(DOCKER_COMPOSE_COMMAND) exec $(PROD_SERVICE_NAME)

build-for-dependencies:
	rm -f *.lock
	$(DOCKER_COMPOSE_COMMAND) build $(SERVICE_NAME)

## Lock dependencies with poetry
lock-dependencies: build-for-dependencies
	$(DOCKER_COMPOSE_RUN) bash -c "if [ -e /home/$(USER_NAME)/poetry.lock.build ]; then cp /home/$(USER_NAME)/poetry.lock.build ./poetry.lock; else poetry lock; fi"

up: 
ifeq (, $(shell docker ps -a | grep $(CONTAINER_NAME)))
	@make down
endif
	$(DOCKER_COMPOSE_COMMAND) --profile $(PROFILE) up -d --remove-orphans


up-prod: 
ifeq (, $(shell docker ps -a | grep $(PROD_CONTAINER_NAME)))
	@make down
endif
	$(DOCKER_COMPOSE_COMMAND) --profile $(PROD_PROFILE) up -d --remove-orphans


down:
	$(DOCKER_COMPOSE_COMMAND) down

exec-in: up
	docker exec -it $(CONTAINER_NAME) bash

prod-exec-in: up-prod
	docker exec -it $(PROD_CONTAINER_NAME) bash

run-prod: up-prod
	$(PROD_DOCKER_COMPOSE_EXEC) python metastatic/run_training.py -m +experiment=sweep_no_smote

run-local: up
	$(DOCKER_COMPOSE_EXEC) python metastatic/run_training.py -m +experiment=sweep_no_smote
	$(DOCKER_COMPOSE_EXEC) python metastatic/run_training.py -m +experiment=sweep_with_smote_lr

build:
	$(DOCKER_COMPOSE_COMMAND) build $(SERVICE_NAME)

notebook: up
	$(DOCKER_COMPOSE_EXEC) jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser	

