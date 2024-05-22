#!/bin/bash

#!/bin/bash
. /opt/conda/bin/activate /opt/conda/envs/flows
sh /opt/conda/bin/activate /opt/conda/envs/flows
conda activate flows
pip install transformers==4.28.1 ml_collections fschat==0.2.20
export WANDB_MODE=disabled

# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'

export model=$1 # llama2 or vicuna
export setup=$2 # behaviors or strings

# Create results folder if it doesn't exist
if [ ! -d "../results" ]; then
    mkdir "../results"
    echo "Folder '../results' created."
else
    echo "Folder '../results' already exists."
fi
#bash run_gcg_individual.sh guanaco behaviors 57 43
for data_offset in $3 #10 20 30 40 50 60 70 80 90
do

    python -u ../main.py \
        --config="../configs/individual_${model}.py" \
        --config.attack=gcg \
        --config.train_data="../../data/advbench/harmful_${setup}.csv" \
        --config.result_prefix="../results/individual_${setup}_${model}_gcg_offset${data_offset}" \
        --config.n_train_data=$4 \
        --config.data_offset=$data_offset \
        --config.test_steps=50 \
        --config.name=${model}
        #--config.batch_size=512

done