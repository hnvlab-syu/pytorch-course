#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/train.py +trainer.precision='16-mixed' trainer.max_epochs=30 trainer=gpu model.net.model="resnet" seed=36 data.batch_size=128 callbacks.early_stopping.patience=5
python src/train.py +trainer.precision='16-mixed' trainer.max_epochs=30 trainer=gpu model.net.model="efficientnet" seed=36 data.batch_size=128 callbacks.early_stopping.patience=5

