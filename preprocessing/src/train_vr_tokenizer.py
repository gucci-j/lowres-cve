import copy
import json
from pathlib import Path

from datasets import load_dataset
from tokenizers.models import BPE
from transformers import AutoTokenizer


def main(args):
    # load the source tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.hub_cache_dir,
    )
    vocab = tokenizer.get_vocab()
    tokenizer_json = json.loads(tokenizer._tokenizer.to_str())
    merges = tokenizer_json["model"]["merges"]
    if tokenizer_json["model"].get("byte_fallback") is not None:
        byte_fallback = tokenizer_json["model"]["byte_fallback"]
    else:
        byte_fallback = False
    if tokenizer_json["model"].get("fuse_unk") is not None:
        fuse_unk = tokenizer_json["model"]["fuse_unk"]
    else:
        fuse_unk = False

    # generate the new tokenizer
    dataset = load_dataset(
        "text", 
        data_files={"train": args.corpus_path},
        cache_dir=args.datasets_cache_dir,
        split="train"
    )
    aux_tokenizer = tokenizer.train_new_from_iterator(
        dataset["text"], args.vocab_size,
    )

    # save
    aux_tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", 
        type=str,
        help="Name or path of the source tokenizer to use",
        required=True
    )
    parser.add_argument(
        "--corpus_path", 
        type=str,
        help="Path to the corpus to train the aux tokenizer on",
        required=True
    )
    parser.add_argument(
        "--vocab_size", 
        type=int,
        help="Vocabulary size of the aux tokenizer",
        required=True
    )
    parser.add_argument(
        "--output_dir", 
        type=str,
        help="Path to the output directory",
        required=True
    )
    parser.add_argument(
        "--lang_code", 
        type=str,
        help="Language code",
        required=True,
        choices=["si", "my", "te"]
    )
    parser.add_argument(
        "--num_new_tokens", 
        type=int,
        help="Number of new tokens to add to the source tokenizer",
        default=100
    )
    parser.add_argument(
        "--datasets_cache_dir", 
        type=str,
        help="Path to the datasets cache directory",
    )
    parser.add_argument(
        "--hub_cache_dir", 
        type=str,
        help="Path to the Hugging Face hub cache directory",
    )
    args = parser.parse_args()
    main(args)
    