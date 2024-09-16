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
    "ar"
    "my"
    "de"
    "el"
    "hi"
    "ja"
    "si"
    "sw"
    "te"
    "th"
)
methods=(
    "align"
    "mean"
    "baseline"
    "merge"
    "focus"
)

for lang_code in "${lang_codes[@]}"; do
    for method in "${methods[@]}"; do
        # align
        if [ "$method" == "align" ]; then
            python main.py \
                --source_model_name_or_path meta-llama/Llama-2-7b-hf \
                --target_tokenizer_name_or_path ${tokenizer_base_dir}/Llama-2-7b-hf-${lang_code}-30K/ \
                --output_dir ${model_base_dir}/Llama-2-7b-hf-${lang_code}-30K-align \
                --cache_dir ${cache_dir} \
                --method align \
                --dataset_path ${cc_100_extracted_file_dir}/${lang_code}_30K.txt \
                --proj
        
        # mean, random, merge
        elif [ "$method" == "mean" ] || [ "$method" == "baseline" ] || [ "$method" == "merge" ]; then
            if [ "$method" == "mean" ]; then
                method="mean"
            elif [ "$method" == "baseline" ]; then
                method="rand"
            elif [ "$method" == "merge" ]; then
                method="merge"
            fi
            python main.py \
                --source_model_name_or_path meta-llama/Llama-2-7b-hf \
                --target_tokenizer_name_or_path ${tokenizer_base_dir}/Llama-2-7b-hf-${lang_code}-30K/ \
                --output_dir ${model_base_dir}/Llama-2-7b-hf-${lang_code}-30K-${method} \
                --cache_dir ${cache_dir} \
                --method ${method} \
                --proj

        # focus
        elif [ "$method" == "focus" ]; then
            python main.py \
                --source_model_name_or_path meta-llama/Llama-2-7b-hf \
                --target_tokenizer_name_or_path ${tokenizer_base_dir}/Llama-2-7b-hf-${lang_code}-30K/ \
                --output_dir ${model_base_dir}/Llama-2-7b-hf-${lang_code}-30K-focus \
                --cache_dir ${cache_dir} \
                --method ${method} \
                --proj \
                --fasttext_model_path ${model_base_dir}/fasttext_model_${lang_code}_30K.bin

        # Invalid method
        else
            echo "Invalid method: ${method}"
        fi
    done
done