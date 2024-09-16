import logging

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core import (instantiate_model_by_merge,
                  instantiate_model_by_align,
                  instantiate_model_by_random,
                  instantiate_model_by_mean,
                  instantiate_model_by_focus)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)


def main(args):
    source_tokenizer = AutoTokenizer.from_pretrained(
        args.source_model_name_or_path,
        cache_dir=args.cache_dir
    )
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_tokenizer_name_or_path)
    source_model = AutoModelForCausalLM.from_pretrained(
        args.source_model_name_or_path,
        cache_dir=args.cache_dir
    )
    
    if args.method == "merge":
        target_model, target_tokenizer = instantiate_model_by_merge(
            source_model, source_tokenizer, target_tokenizer, not args.proj
        )
    elif args.method == "align":
        target_model, target_tokenizer = instantiate_model_by_align(
            source_model, source_tokenizer, target_tokenizer, 
            args.dataset_path, not args.proj, 
            args.use_only_merge_for_head, args.use_only_merge_for_embeddings,
            args.use_only_align, args.consider_mean
        )
    elif args.method == "baseline":
        target_model, target_tokenizer = instantiate_model_by_random(
            source_model, source_tokenizer, target_tokenizer, not args.proj
        )
    elif args.method == "mean":
        target_model, target_tokenizer = instantiate_model_by_mean(
            source_model, source_tokenizer, target_tokenizer, not args.proj
        )
    elif args.method == "focus":
        target_model, target_tokenizer = instantiate_model_by_focus(
            source_model, source_tokenizer, target_tokenizer, 
            args.fasttext_model_path, not args.proj, args.is_word_level
        )
    
    # Save the target model and tokenizer
    target_model.save_pretrained(args.output_dir)
    target_tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser("Initialize the target model.")
    parser.add_argument(
        "--source_model_name_or_path", 
        type=str, 
        required=True,
        help="The source model to initialize the target model with."
    )
    parser.add_argument(
        "--target_tokenizer_name_or_path", 
        type=str, 
        required=True,
        help="The target tokenizer to initialize the target model with."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="The output directory to save the target model and tokenizer."
    )
    parser.add_argument(
        "--cache_dir", 
        type=str, 
        default=None,
        help="The cache directory to save the source model and tokenizer."
    )
    parser.add_argument(
        "--proj", 
        action="store_true",
        help="Whether to apply the output projection init."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="merge",
        choices=["merge", "align", "baseline", "mean", "focus"],
        help="The method to initialize the target model."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="[align] The path to the dataset for aligning the target tokenizer."
    )
    parser.add_argument(
        "--use_only_merge_for_head",
        action="store_true",
        help="[align] Whether to use only merge-based init for an LM head."
    )
    parser.add_argument(
        "--use_only_merge_for_embeddings",
        action="store_true",
        help="[align] Whether to use only merge-based init for embeddings."
    )
    parser.add_argument(
        "--use_only_align",
        action="store_true",
        help="[align] Whether to use only align-based init."
    )
    parser.add_argument(
        "--consider_mean",
        action="store_true",
        help="[align] Whether to consider the mean of embeddings for align-based init."
    )
    parser.add_argument(
        "--fasttext_model_path",
        type=str,
        default=None,
        help="[focus] The path to the FastText model."
    )
    parser.add_argument(
        "--is_word_level",
        action="store_true",
        help="[focus] Whether to use word-level embeddings."
    )
    args = parser.parse_args()
    main(args)
