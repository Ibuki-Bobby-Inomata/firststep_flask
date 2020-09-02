.PHONY: clean clean-model clean-pyc docs help init init-docker create-container start-container jupyter test lint profile clean clean-data clean-docker clean-container clean-image sync-from-source sync-to-source
.DEFAULT_GOAL := help

###########################################################################################################
## SCRIPTS
###########################################################################################################

define PRINT_HELP_PYSCRIPT
import os, re, sys

if os.environ['TARGET']:
    target = os.environ['TARGET']
    is_in_target = False
    for line in sys.stdin:
        match = re.match(r'^(?P<target>{}):(?P<dependencies>.*)?## (?P<description>.*)$$'.format(target).format(target), line)
        if match:
            print("target: %-20s" % (match.group("target")))
            if "dependencies" in match.groupdict().keys():
                print("dependencies: %-20s" % (match.group("dependencies")))
            if "description" in match.groupdict().keys():
                print("description: %-20s" % (match.group("description")))
            is_in_target = True
        elif is_in_target == True:
            match = re.match(r'^\t(.+)', line)
            if match:
                command = match.groups()
                print("command: %s" % (command))
            else:
                is_in_target = False
else:
    for line in sys.stdin:
        match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
        if match:
            target, help = match.groups()
            print("%-20s %s" % (target, help))
endef

define START_DOCKER_CONTAINER
if [ `$(DOCKER) inspect -f {{.State.Running}} $(CONTAINER_NAME)` = "false" ] ; then
        $(DOCKER) start $(CONTAINER_NAME)
fi
endef

###########################################################################################################
## VARIABLES
###########################################################################################################
export DOCKER=docker
export TARGET=
export PRINT_HELP_PYSCRIPT
export START_DOCKER_CONTAINER
export PYTHONPATH=$$PYTHONPATH:$(PWD)
export PROJECT_NAME=flask_step
export DOCKER_USER=ibukiinomata
export IMAGE_NAME=$(DOCKER_USER)/$(PROJECT_NAME)-image
export CONTAINER_NAME=$(PROJECT_NAME)-container
export JUPYTER_HOST_PORT=9010
export JUPYTER_CONTAINER_PORT=9010
export FLASK_HOST_PORT=5030
export FLASK_CONTAINER_PORT=5030
export CODE_HOST_PORT=8443
export CODE_CONTAINER_PORT=8443
export PYTHON=python3
export DOCKERFILE=docker/Dockerfile

###########################################################################################################
## ADD TARGETS SPECIFIC TO "beacon-near2"
###########################################################################################################


###########################################################################################################
## GENERAL TARGETS
###########################################################################################################

help: ## show this message
	@$(PYTHON) -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

init-docker: ## initialize docker image
	$(DOCKER) build --no-cache -t $(IMAGE_NAME) -f $(DOCKERFILE) .

create-image:
	$(DOCKER) build -t $(IMAGE_NAME) -f $(DOCKERFILE) .

create-container: ## create docker container
	$(DOCKER) run -it -v $(PWD):/work -p $(JUPYTER_HOST_PORT):$(JUPYTER_CONTAINER_PORT) -p $(FLASK_HOST_PORT):$(FLASK_CONTAINER_PORT)  --name $(CONTAINER_NAME) $(IMAGE_NAME)

create-app-container: ## create docker container without port forwarding
	$(DOCKER) run -it -v $(PWD):/work --name $(CONTAINER_NAME) $(IMAGE_NAME)

app:
	$(PYTHON) web/app.py

start-container: ## start docker container
	@echo "$$START_DOCKER_CONTAINER" | $(SHELL)
	@echo "Launched $(CONTAINER_NAME)..."
	$(DOCKER) attach $(CONTAINER_NAME)

lab:
	jupyter lab --allow-root &

push-image:
	$(DOCKER) tag $(IMAGE_NAME) registry-sd.com/$(DOCKER_USER)/$(IMAGE_NAME)
	$(DOCKER) push $(IMAGE_NAME)

jupyter: ## start Jupyter Notebook server
	jupyter lab --ip=0.0.0.0 --port=${JUPYTER_CONTAINER_PORT} --allow-root

code-server:
	docker run -p $(CODE_HOST_PORT):$(CODE_CONTAINER_PORT) -v "$(PWD):/root/project" codercom/code-server code-server --allow-http --no-auth

test: ## run test cases in tests directory
	$(PYTHON) -m unittest discover

lint: ## check style with flake8
	flake8 beacon_near2

jupyter-lab:
	jupyter lab --allow-root &

profile: ## show profile of the project
	@echo "CONTAINER_NAME: $(CONTAINER_NAME)"
	@echo "IMAGE_NAME: $(IMAGE_NAME)"
	@echo "JUPYTER_PORT: `$(DOCKER) port $(CONTAINER_NAME)`"
	@echo "DATA_SOURE: $(DATA_SOURCE)"

clean: clean-model clean-pyc clean-docker ## remove all artifacts

clean-model: ## remove model artifacts
	rm -fr model/*

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

distclean: clean clean-data ## remove all the reproducible resources including Docker images

clean-data: ## remove files under data
	rm -fr data/*

clean-docker: clean-container clean-image ## remove Docker image and container

clean-container: ## remove Docker container
	-$(DOCKER) rm $(CONTAINER_NAME)

clean-image: ## remove Docker image
	-$(DOCKER) image rm $(IMAGE_NAME)
