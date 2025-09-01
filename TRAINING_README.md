# HRM Training Setup

minimal training infrastructure for the HRM model on ARC data.

## setup

```bash
# build image
make build

# start container
make run
```

## usage

### 1. preprocess data

convert ARC JSON files to tensor format:

```bash
make preprocess
```

this creates:
- `data/train/` - 80% of data for training
- `data/val/` - 20% of data for validation

### 2. train model

```bash
make train
```

training will:
- load config from `configs/hrm_default.yaml`
- save checkpoints to `checkpoints/`
- print metrics each epoch
- early stop on validation perfect-grid@last

### 3. evaluate model

```bash
make evaluate
```

loads best checkpoint and evaluates on validation set.

## docker commands

```bash
make build      # build image
make run        # start container + shell
make stop       # stop container
make shell      # open shell in running container
make preprocess # run preprocessing
make train      # run training
make evaluate   # run evaluation
make clean      # clean up containers and images
```

## configuration

edit `configs/hrm_default.yaml` for model/training parameters.

edit `configs/augmentation.yaml` for augmentation settings.

## data flow

```
ARC-AGI-2/data/training/*.json (read-only)
    ↓
scripts/preprocess.py
    ↓
data/train/ + data/val/ (tensor files)
    ↓
scripts/train.py (training loop)
    ↓
checkpoints/ (model saves)
```

## expected behavior

- early epochs: fast CE drop, cell-acc grows
- mid training: perfect-grid@last climbs steadily
- early stopping: patience 15 epochs on val perfect-grid@last
- checkpointing: saves best by validation perfect-grid@last

## gpu requirements

- nvidia-docker2 installed
- CUDA 12.1+ compatible GPU
- docker compose (modern syntax)