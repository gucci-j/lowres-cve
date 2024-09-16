#!/bin/bash

#####
# parameters
#####
# You just need to change these paths
custom_task_script_dir="/path/to/eval/src"
log_base_dir="/path/to/log/dir"
cache_dir="/path/to/cache/dir"
light_eval_dir="/path/to/lighteval"
model_base_dir=/path/to/models
checkpoint_path="/path/to/BLEURT-20"

model_identifier="Meta-Llama-3-8B"
source_model_name="meta-llama/Meta-Llama-3-8B"
model_abbrev="llama3"

lang_codes=(
    "si"
    "te"
    "my"
)

for lang_code in "${lang_codes[@]}"; do
    model_paths=(
        "${model_base_dir}/${model_identifier}-${lang_code}-30K-align-tuned"
        "${model_base_dir}/${model_identifier}-${lang_code}-30K-mean-tuned"
        "${model_base_dir}/${model_identifier}-${lang_code}-30K-rand-tuned"
    )

    tasks=(
        "custom|mt:en2${lang_code}|0|0"
        "custom|mt:en2${lang_code}|0|0"
        "custom|mt:en2${lang_code}|0|0"
        "custom|mt:en2${lang_code}|0|0"
        "custom|mt:en2${lang_code}|0|0"
        "custom|sum:${lang_code}|0|1"
        "custom|sum:${lang_code}|0|1"
        "custom|sum:${lang_code}|0|1"
        "custom|sum:${lang_code}|0|1"
        "custom|sum:${lang_code}|0|1"
    )

    #####
    # Edit config file and copy tokenizer
    #####
    for model_path in "${model_paths[@]}"; do
        model_name=$(basename ${model_path})

        # Edit config.json
        echo "Editing config.json..."
        config_path="${model_base_dir}/${model_name}/config.json"
        sed -i "s|\"torch_dtype\": \".*\"|\"torch_dtype\": \"float16\"|" ${config_path}

        echo "Copying tokenizer..."
        temp_model_name=$(echo $model_name | sed 's/\(-proj\).*$/\1/')
        tokenizer_path="${model_base_dir}/${temp_model_name}"
        cp ${tokenizer_path}/tokenizer* ${model_path}
        cp ${tokenizer_path}/special_tokens_map.json ${model_path}
    done


    #####
    # Run evaluations
    #####
    cd ${light_eval_dir}
    model_paths=(
        "${model_base_dir}/${model_identifier}-${lang_code}-30K-align-tuned"
        "${model_base_dir}/${model_identifier}-${lang_code}-30K-mean-tuned"
        "${model_base_dir}/${model_identifier}-${lang_code}-30K-rand-tuned"
        ${source_model_name}
    )
    for model_path in "${model_paths[@]}"; do
        model_name=$(basename ${model_path})
        echo "Evaluating model ${model_name}..."
        
        for task in "${tasks[@]}"; do
            # get a task name with out lang_code
            task_name=$(echo $task | cut -d'|' -f2 | cut -d':' -f1)

            accelerate launch --mixed_precision=bf16 run_evals_accelerate.py \
                --model_args "pretrained=${model_path}" \
                --tasks "${task}" \
                --custom_tasks "${custom_task_script_dir}/${task_name}.py" \
                --override_batch_size 1 \
                --output_dir="${log_base_dir}/${model_abbrev}/${task_name}" \
                --cache_dir="${cache_dir}"

        done
    done

    #####
    # Compute BLUERT scores
    #####
    cd ${custom_task_script_dir}
    python compute_bleurt.py \
        --parquet_dir "${log_base_dir}/${model_abbrev}/sum/details" \
        --checkpoint_path ${checkpoint_path}

done
