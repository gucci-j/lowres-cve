#!/bin/bash

# Configs
export CUDA_HOME=/usr/local/cuda-12.9
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TMPDIR=/tmp
export TRANSFORMERS_VERBOSITY=debug
export HF_HOME=/path/to/cache
export HF_HUB_CACHE=/path/to/cache
export HF_DATASETS_CACHE=/path/to/cache
export HF_DATASETS_TRUST_REMOTE_CODE=true
model_name_or_path="$1"
model_abbrev=$(cut -d'/' -f2 <<< $model_name_or_path)
lang_code="$2"
init="$3"
model_name_or_path="/path/to/models/${model_abbrev}-${lang_code}-vr-${init}" # Specify model path generated in the initialization phase
lapt_data_dir="/path/to/datasets"
model_base_dir="/path/to/models"
logging_dir="/path/to/lowres-cve/lapt/logs"
if [[ "${model_abbrev}" == "gemma-2-9b" ]]; then
    model_type="gemma2"
else
    echo "Unsupported model abbreviation: ${model_abbrev}"
    exit 1
fi

cd /path/to/lowres-cve/lapt/src
python main_vr.py \
    --dataset_path ${lapt_data_dir}/${model_abbrev}-${lang_code}-vr \
    --output_dir ${model_base_dir}/${model_abbrev}-${lang_code}-vr-${init}-tuned \
    --logging_dir ${logging_dir}/${model_abbrev}-${lang_code}-vr-${init} \
    --model_name_or_path ${model_name_or_path} \
    --tokenizer_name_or_path ${model_name_or_path} \
    --model_type ${model_type} \
    --seed 42 \
    --eval_strategy no \
    --logging_steps 5 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --prediction_loss_only \
    --overwrite_output_dir \
    --do_train \
    --lr_scheduler_type cosine \
    --disable_tqdm True \
    --label_names labels \
    --remove_unused_columns False \
    --save_strategy epoch \
    --bf16 \
    --gradient_checkpointing True
