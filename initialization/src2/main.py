import logging

from transformers import AutoModelForCausalLM, AutoTokenizer

from methods import (
    instantiate_model_by_random,
    instantiate_model_by_mean,
    instantiate_model_by_focus,
)

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
    
    if args.method == "random":
        target_model, target_tokenizer = instantiate_model_by_random(
            source_model, target_tokenizer, source_model.config.tie_word_embeddings
        )
    elif args.method == "mean":
        target_model, target_tokenizer = instantiate_model_by_mean(
            source_model, source_tokenizer, target_tokenizer, source_model.config.tie_word_embeddings
        )
    elif args.method == "focus":
        target_model, target_tokenizer = instantiate_model_by_focus(
            source_model, source_tokenizer, target_tokenizer, 
            args.fasttext_model_path, source_model.config.tie_word_embeddings
        )
    else:
        raise ValueError(f"Invalid method: {args.method}")
    
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
        "--method",
        type=str,
        default="merge",
        choices=["random", "mean", "focus"],
        help="The method to initialize the target model."
    )
    parser.add_argument(
        "--fasttext_model_path",
        type=str,
        default=None,
        help="[focus] The path to the FastText model."
    )
    args = parser.parse_args()
    main(args)
