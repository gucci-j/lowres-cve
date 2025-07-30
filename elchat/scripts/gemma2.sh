#!/bin/bash

source $PROJ_HOME/envs/lowres-cve/bin/activate

# Configs
export CUDA_HOME=/usr/local/cuda-12.9
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TMPDIR=/tmp
export TRANSFORMERS_VERBOSITY=debug
export HF_HOME="/path/to/cache/cache"
export HF_HUB_CACHE="/path/to/cache/cache"
export HF_DATASETS_CACHE="/path/to/cache/cache"
export HF_DATASETS_TRUST_REMOTE_CODE=true
cache_dir="/path/to/cache/cache"
model_abbrev="gemma-2-9b"
lang_code="$1"

# Run the script
cd /path/to/lowres-cve/elchat/src
python main.py \
    --model_src_name_or_path "atsuki-yamaguchi/gemma-2-9b-${lang_code}-30K-align" \
    --model_tgt_name_or_path "google/gemma-2-9b-it" \
    --tokenizer_src_name_or_path "atsuki-yamaguchi/gemma-2-9b-${lang_code}-30K-align" \
    --pipeline add_transition copy_emb \
    --consider_special_tokens \
    --transition_indices 0 1 -2 -1 \
    --transition_rates 0.3 0.5 0.5 0.3 \
    --transition_method slerp \
    --cache_dir "${cache_dir}" \
    --output_dir "/path/to/models/${model_abbrev}-${lang_code}-30K-align-merge"
