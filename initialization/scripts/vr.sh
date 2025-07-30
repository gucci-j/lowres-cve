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
tokenizer_name_or_path="/path/to/tokenizers/${model_abbrev}-${lang_code}/" # Specify output path in train_vr_tokenizer.sh
cache_dir="/path/to/cache"



# Run the script
cd /path/to/lowres-cve/initialization/src2 # Make sure to set the correct path. It is not "src" but "src2" as "src" is used for vocabulary expansion.

output_dir="/path/to/output/dir/${model_abbrev}-${lang_code}-vr-mean"
mkdir -p $output_dir
python main.py \
    --source_model_name_or_path ${model_name_or_path} \
    --target_tokenizer_name_or_path ${tokenizer_name_or_path} \
    --output_dir ${output_dir} \
    --cache_dir ${cache_dir} \
    --method "mean"

output_dir="/path/to/output/dir/${model_abbrev}-${lang_code}-vr-random"
mkdir -p $output_dir
python main.py \
    --source_model_name_or_path ${model_name_or_path} \
    --target_tokenizer_name_or_path ${tokenizer_name_or_path} \
    --output_dir ${output_dir} \
    --cache_dir ${cache_dir} \
    --method "random"

output_dir="/path/to/output/dir/${model_abbrev}-${lang_code}-vr-focus"
mkdir -p $output_dir
python main.py \
    --source_model_name_or_path ${model_name_or_path} \
    --target_tokenizer_name_or_path ${tokenizer_name_or_path} \
    --output_dir ${output_dir} \
    --cache_dir ${cache_dir} \
    --method "focus"
