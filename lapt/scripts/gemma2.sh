#!/bin/bash

#####
# Parameters
#####
# You just need to change these paths
model_base_dir=/path/to/models
lapt_data_dir=/path/to/output

lang_codes=(
    "si"
    "te"
    "my"
)

methods=(
    "align"
    "mean"
    "baseline"
)

for lang_code in "${lang_codes[@]}"; do
    for method in "${methods[@]}"; do
        if [ "$method" == "baseline" ]; then
            method="rand"
        else
            method="${method}"
        fi

        python main_2x2ls.py \
            --dataset_path ${lapt_data_dir}/cc100_${lang_code}_30K_gemma2 \
            --output_dir ${model_base_dir}/gemma-2-9b-${lang_code}-30K-${method}-tuned/ \
            --logging_dir ${model_base_dir}/gemma-2-9b-${lang_code}-30K-${method}-tuned/ \
            --model_name_or_path ${model_base_dir}/gemma-2-9b-${lang_code}-30K-${method} \
            --tokenizer_name_or_path ${model_base_dir}/gemma-2-9b-${lang_code}-30K-${method} \
            --model_type llama2 \
            --seed 42 \
            --evaluation_strategy no \
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
            --gradient_checkpointing True \
            --tune_embeddings \
            --copy_lm_head

    done
done
