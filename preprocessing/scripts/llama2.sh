#!/bin/bash

cd /path/to/preprocessing/src

#####
# Parameters
#####
# You just need to change these paths
tokenizer_base_dir=/path/to/tokenizers
model_base_dir="/path/to/models"
cc100_file_base_dir="/path/to/cc100"
source_tokenizer_path=/path/to/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852/tokenizer.model
lapt_data_dir="/path/to/output"

vocab_size=50000
num_sentences=30000
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

for lang_code in "${lang_codes[@]}"; do
    #####
    # 1. Set up
    #####
    tokenizer_dir=${tokenizer_base_dir}/Llama-2-7b-hf-${lang_code}-30K/
    mkdir $tokenizer_dir
    cc100_file_path="${cc100_file_base_dir}/${lang_code}.txt"
    cc100_extracted_file_path="${lapt_data_dir}/${lang_code}_30K.txt"
    ppl_extracted_file_path="${lapt_data_dir}/${lang_code}_ppl_data.txt"
    shuf -n "$num_sentences" "$cc100_file_path" > "$cc100_extracted_file_path"

    #####
    # 2. Generate perplexity held-out data
    #####
    python extract_ppl_data.py \
        --source_file_path ${cc100_file_path} \
        --target_file_path_1 ${cc100_extracted_file_path} \
        --output_file_path ${ppl_extracted_file_path}

    #####
    # 3. Train tokenizer
    #####
    if [ "$lang_code" == "si" ] || [ "$lang_code" == "my" ] || [ "$lang_code" == "te" ]; then
        # Using HF tokenizers for better results in my, si, and te.
        python train_tokenizer_llama2.py \
            --corpus_path ${cc100_extracted_file_path} \
            --vocab_size ${vocab_size} \
            --output_dir ${tokenizer_dir} \
            --lang_code ${lang_code} \
            --num_new_tokens 100
    else
        python train_tokenizer_with_filtering_llama2.py \
            --corpus_path ${cc100_extracted_file_path} \
            --vocab_size ${vocab_size} \
            --source_tokenizer_path ${source_tokenizer_path} \
            --output_dir ${tokenizer_dir} \
            --lang_code ${lang_code} \
            --num_new_tokens 100
    fi

    #####
    # 4. Generate training and evaluation data
    #####
    python generate_lapt_data.py \
        --data_path ${cc100_extracted_file_path} \
        --output_data_path "${lapt_data_dir}/cc100_${lang_code}_30K" \
        --tokenizer_name_or_path ${tokenizer_dir} \
        --num_workers 4 \
        --max_length 512

    python generate_lapt_data.py \
        --data_path ${ppl_extracted_file_path} \
        --output_data_path "${lapt_data_dir}/ppl_${lang_code}_30K" \
        --tokenizer_name_or_path ${tokenizer_dir} \
        --num_workers 4 \
        --max_length 2048

    #####
    # 5. Train fastText embeddings (optional)
    #####
    input_file="${cc100_extracted_file_path}"
    python train_fasttext.py \
        --tokenizer_name_or_path ${tokenizer_dir} \
        --text_path ${input_file} \
        --min_length 5 \
        --target_lang ${lang_code} \
        --model_cache_dir ${model_base_dir}
    
done
