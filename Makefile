.PHONY: build run stop shell preprocess train evaluate streamlit help

# default target
help:
	@echo "available commands:"
	@echo "  make build      - build docker image"
	@echo "  make run        - start container and open shell"
	@echo "  make stop       - stop container"
	@echo "  make shell      - open shell in running container"
	@echo "  make preprocess - run data preprocessing for both agi1 and agi2"
	@echo "  make train      - run training on agi2 (default)"
	@echo "  make train dataset=agi1 - run training on agi1"
	@echo "  make evaluate   - run evaluation on agi2 (default)"
	@echo "  make evaluate dataset=agi1 - run evaluation on agi1"
	@echo "  make streamlit  - start streamlit testing interface"
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

# run preprocessing (both datasets)
preprocess:
	docker compose exec hrm-training python scripts/preprocess.py

# run training (default: agi2)
train:
	docker compose exec hrm-training python scripts/train.py --dataset $(or $(dataset),agi2)

# run evaluation (default: agi2)
evaluate:
	docker compose exec hrm-training python scripts/evaluate.py --dataset $(or $(dataset),agi2)

# start streamlit testing interface
streamlit:
	docker compose up hrm-streamlit

# clean up
clean:
	docker compose down -v
	docker system prune -f 

lint:
	ruff check --fix . --exclude HRM,ARC-AGI,ARC-AGI2
	ruff format --exclude HRM,ARC-AGI,ARC-AGI2