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
cache_dir="/path/to/cache"
tokenizer_name_or_path="/path/to/tokenizers/${model_abbrev}-${lang_code}" # Specify output path in train_vr_tokenizer.sh
data_dir="/path/to/cc100/data"
text_path="${data_dir}/${lang_code}_30K.txt"
output_dir="/path/to/output/dir"

# Run the script
cd /path/to/lowres-cve/preprocessing/src
mkdir -p $output_dir
python train_vr_fasttext.py \
    --tokenizer_name_or_path ${tokenizer_name_or_path} \
    --cache_dir ${cache_dir} \
    --text_path ${text_path} \
    --lang_code ${lang_code} \
    --data_dir ${data_dir} \
    --output_dir ${output_dir} \
    --model_abbrev ${model_abbrev}
