#!/bin/bash

cd /path/to/preprocessing/src

#####
# Parameters
#####
# You just need to change these paths
tokenizer_base_dir=/path/to/tokenizers
model_base_dir="/path/to/models"
cc100_file_base_dir="/path/to/cc100"
source_tokenizer_path=/path/to/hub/models--google--gemma-2-9b/snapshots/33c193028431c2fde6c6e51f29e6f17b60cbfac6/tokenizer.model
lapt_data_dir="/path/to/output"

vocab_size=50000
num_sentences=30000

lang_codes=(
    "my"
    "si"
    "te"
)

for lang_code in "${lang_codes[@]}"; do
    #####
    # 1. Set up
    #####
    tokenizer_dir=${tokenizer_base_dir}/gemma-2-9b-${lang_code}-30K/
    mkdir $tokenizer_dir
    cc100_file_path="${cc100_file_base_dir}/${lang_code}.txt"
    cc100_extracted_file_path="${lapt_data_dir}/${lang_code}_30K.txt"
    ppl_extracted_file_path="${lapt_data_dir}/${lang_code}_ppl_data.txt"

    #####
    # 2. Train tokenizer
    #####
    python train_tokenizer_with_filtering_gemma2.py \
        --corpus_path ${cc100_extracted_file_path} \
        --vocab_size ${vocab_size} \
        --source_tokenizer_path ${source_tokenizer_path} \
        --output_dir ${tokenizer_dir} \
        --lang_code ${lang_code} \
        --num_new_tokens 100

    #####
    # 3. Generate training and evaluation data
    #####
    python generate_lapt_data.py \
        --data_path ${cc100_extracted_file_path} \
        --output_data_path "${lapt_data_dir}/cc100_${lang_code}_30K_gemma2" \
        --tokenizer_name_or_path ${tokenizer_dir} \
        --num_workers 4 \
        --max_length 512

    python generate_lapt_data.py \
        --data_path ${ppl_extracted_file_path} \
        --output_data_path "${lapt_data_dir}/ppl_${lang_code}_30K_gemma2" \
        --tokenizer_name_or_path ${tokenizer_dir} \
        --num_workers 4 \
        --max_length 2048

done
