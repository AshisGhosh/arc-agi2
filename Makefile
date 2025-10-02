.PHONY: build run stop shell preprocess train evaluate help

# default target
help:
	@echo "available commands:"
	@echo "  make build      - build docker image"
	@echo "  make run        - start container and open shell"
	@echo "  make stop       - stop container"
	@echo "  make shell      - open shell in running container"
	@echo "  make preprocess - run data preprocessing"
	@echo "  make train      - run training"
	@echo "  make evaluate   - run evaluation"
	@echo "  make view       - launch dataset viewer"
	@echo "  make view-model - launch model predictions viewer"
	@echo "  make analyze    - launch detailed model analysis"
	@echo "  make clean      - clean up containers and images"
	@echo "  make lint       - run linting"

# build docker image
build:
	docker compose build

# start container and open shell
run:
	docker compose up -d
	docker compose exec hrm-training bash

# stop container
stop:
	docker compose down

# open shell in running container
shell:
	docker compose exec hrm-training bash

# run preprocessing
preprocess:
	docker compose exec hrm-training python scripts/preprocess.py

# run training
DATA ?= arc1
train:
	docker compose exec hrm-training python scripts/train.py $(DATA)

train-overfit:
	docker compose exec hrm-training python scripts/overfit_experiment.py

# run evaluation
evaluate:
	docker compose exec hrm-training python scripts/evaluate.py

# launch dataset viewer
view:
	docker compose exec hrm-training streamlit run scripts/view_dataset.py --server.port 8502 --server.address 0.0.0.0

# launch model predictions viewer
view-model:
	docker compose exec hrm-training streamlit run scripts/view_model_predictions.py --server.port 8503 --server.address 0.0.0.0

# launch detailed model analysis
analyze:
	docker compose exec hrm-training streamlit run scripts/detailed_analysis_app.py --server.port 8504 --server.address 0.0.0.0

# clean up
clean:
	docker compose down -v
	docker system prune -f 

lint:
	ruff check --fix . --exclude HRM
	ruff format --exclude HRM