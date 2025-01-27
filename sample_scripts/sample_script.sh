#!/bin/bash

CONDA_ENV="FedUniBrain"
EXPERIMENT_NAME="fedunibrain_atlas_brats"
OUTPUT_FOLDER="../output/"

# Activate conda env if in base env, or don't if already set.
source "$(dirname $(dirname $(which conda)))/etc/profile.d/conda.sh"
if [[ "${CONDA_DEFAULT_ENV}" != "${CONDA_ENV}" ]]; then
  echo "activating ${CONDA_ENV} env"
  set +u; conda activate "${CONDA_ENV}"; set -u
fi

# Create outputfolder if it does not exist
mkdir -p ${OUTPUT_FOLDER}${EXPERIMENT_NAME}

# Run the experiment
python3 ../main_federated.py \
  --experiment_name $EXPERIMENT_NAME \
  --gpu 0 \
  --wandb_project "MT-MRI" \
  --datasets "ATLAS" "BRATS" \
  --evaluation_datasets "ATLAS" "BRATS" \
  --num_workers 8 \
  --equal_weighting \
  --batch_size 8 \
  --norm "BATCH" \
  --aggregation_method "fedbn" \
  --training_algorithm "default" \
  --loss_argument "diceandsoftbce:bce_weight=0.2,dice_weight=0.8,label_smoothing=0.15" \
  --validation_interval 5

conda deactivate