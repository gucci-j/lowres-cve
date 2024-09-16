#!/bin/bash

cd /path/to/instantiation/src

#####
# Parameters
#####
# You just need to change these paths
tokenizer_base_dir=/path/to/tokenizers
model_base_dir=/path/to/models
cc_100_extracted_file_dir=/path/to/output/
cache_dir=/path/to/hub

lang_codes=(
    "my"
    "si"
    "te"
)
methods=(
    "align"
    "mean"
    "baseline"
)

for lang_code in "${lang_codes[@]}"; do
    for method in "${methods[@]}"; do
        # align
        if [ "$method" == "align" ]; then
            python main.py \
                --source_model_name_or_path google/gemma-2-9b \
                --target_tokenizer_name_or_path ${tokenizer_base_dir}/gemma-2-9b-${lang_code}-30K/ \
                --output_dir ${model_base_dir}/gemma-2-9b-${lang_code}-30K-align \
                --cache_dir ${cache_dir} \
                --method align \
                --dataset_path ${cc_100_extracted_file_dir}/${lang_code}_30K.txt
        
        # mean, random, merge
        elif [ "$method" == "mean" ] || [ "$method" == "baseline" ]; then
            if [ "$method" == "mean" ]; then
                method="mean"
            elif [ "$method" == "baseline" ]; then
                method="rand"
            fi
            python main.py \
                --source_model_name_or_path google/gemma-2-9b \
                --target_tokenizer_name_or_path ${tokenizer_base_dir}/gemma-2-9b-${lang_code}-30K/ \
                --output_dir ${model_base_dir}/gemma-2-9b-${lang_code}-30K-${method} \
                --cache_dir ${cache_dir} \
                --method ${method}

        # Invalid method
        else
            echo "Invalid method: ${method}"
        fi
    done
done