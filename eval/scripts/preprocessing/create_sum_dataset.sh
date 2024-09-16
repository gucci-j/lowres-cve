#!/bin/bash

cd /path/to/eval/src/

lang_codes=(
    "ja"
    "de"
    "ar"
    "sw"
    "el"
    "th"
    "hi"
    "my"
    "si"
    "te"
)

for lang_code in "${lang_codes[@]}"; do
    python create_hf_sum_datasets.py \
        --output_dir /path/to/output/dir \
        --cache_dir /path/to/cache/dir \
        --repo_id your-hub-id/sum-${lang_code} \
        --lang_code ${lang_code}
done
