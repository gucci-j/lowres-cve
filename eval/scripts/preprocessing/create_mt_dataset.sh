#!/bin/bash

cd /path/to/eval/src/

python create_hf_mt_datasets.py \
    --data_dir /path/to/floresp-v2.0-alpha.2 \
    --output_dir /path/to/output/dir \
    --cache_dir /path/to/hub \
    --repo_id your-hub-id/flores
