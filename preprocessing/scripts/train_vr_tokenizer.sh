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
data_dir=/path/to/cc100/data
model_name_or_path="$1"
model_abbrev=$(cut -d'/' -f2 <<< $model_name_or_path)
lang_code="$2"
vocab_size=32000
corpus_path="${data_dir}/${lang_code}_30K.txt"
output_dir=/path/to/output/dir/${model_abbrev}-${lang_code}/

# Extract 30K sentences from CC100 dataset if not already done
if [ ! -f "${corpus_path}" ]; then
    echo "Extracting 30K sentences from CC100 dataset for language: ${lang_code}"
    cc100_file_path="${data_dir}/${lang_code}.txt"
    cc100_extracted_file_path="${data_dir}/${lang_code}_30K.txt"
    num_sentences=30000
    shuf -n "$num_sentences" "$cc100_file_path" > "$cc100_extracted_file_path"
    corpus_path="${cc100_extracted_file_path}"
else
    echo "Using existing corpus file: ${corpus_path}"
fi

# Run the script
cd /path/to/lowres-cve/preprocessing/src
mkdir -p $output_dir
python train_vr_tokenizer.py \
    --corpus_path ${corpus_path} \
    --vocab_size ${vocab_size} \
    --output_dir ${output_dir} \
    --lang_code ${lang_code} \
    --datasets_cache_dir "${HF_DATASETS_CACHE}" \
    --hub_cache_dir "${HF_HUB_CACHE}" \
    --model_name_or_path ${model_name_or_path}
