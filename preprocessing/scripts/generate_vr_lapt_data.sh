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
tokenizer_name_or_path="/path/to/tokenizers/${model_abbrev}-${lang_code}" # Specify output path in train_vr_tokenizer.sh
data_dir="/path/to/cc100/data"
cc100_file_path="${data_dir}/${lang_code}.txt"
cc100_extracted_file_path="${data_dir}/${lang_code}_30K.txt"

cd /path/to/lowres-cve/preprocessing/src

python generate_lapt_data.py \
    --data_path ${cc100_extracted_file_path} \
    --output_data_path "${data_dir}/${model_abbrev}-${lang_code}-vr" \
    --tokenizer_name_or_path ${tokenizer_name_or_path} \
    --num_workers 16 \
    --max_length 512
