import copy
import json
from pathlib import Path

from datasets import load_dataset
from tokenizers.models import BPE
from transformers import AutoTokenizer


def main(args):
    # load the source tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        cache_dir=args.hub_cache_dir,
        legacy=False
    )
    vocab = tokenizer.get_vocab()
    tokenizer_json = json.loads(tokenizer._tokenizer.to_str())
    merges = tokenizer_json["model"]["merges"]

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
    aux_tokenizer_json = json.loads(aux_tokenizer._tokenizer.to_str())
    aux_merges = aux_tokenizer_json["model"]["merges"]

    # merge the tokenizers
    num_new_token = 0
    max_new_token = args.num_new_tokens
    ret_vocab = copy.copy(vocab)
    ret_merges = []
    old_merges = copy.copy(merges)
    for merge in aux_merges:
        # vocab
        token_1, token_2 = merge.split(" ")
        token = token_1 + token_2
        if num_new_token < max_new_token:
            if token_1 not in ret_vocab and token_2 not in ret_vocab: # both are new
                ret_vocab[token_1] = len(vocab) + num_new_token
                ret_vocab[token_2] = len(vocab) + num_new_token + 1
                num_new_token += 2
            elif token_1 not in ret_vocab and token_2 in ret_vocab: # new + old
                ret_vocab[token_1] = len(vocab) + num_new_token
                num_new_token += 1
            elif token_1 in ret_vocab and token_2 not in ret_vocab: # old + new
                ret_vocab[token_2] = len(vocab) + num_new_token
                num_new_token += 1
            else: # both are old
                pass
            if token not in ret_vocab:
                ret_vocab[token] = len(vocab) + num_new_token
                num_new_token += 1

        # merge
        if merge in merges:
            old_merges.remove(merge)
            ret_merges.append(merge)
        elif token in ret_vocab and token_1 in ret_vocab and token_2 in ret_vocab:
            ret_merges.append(merge)
    
    # retrain tokenizer
    merges = ret_merges + old_merges
    vocab = ret_vocab
    tokenizer.backend_tokenizer.model = BPE(
        vocab=vocab,
        merges=[(merge.split(' ')[0], merge.split(' ')[1]) for merge in merges],
        fuse_unk=False,
        byte_fallback=True,
    )

    # save
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus_path", 
        type=str,
        help="Path to the corpus to train the tokenizer on",
        required=True
    )
    parser.add_argument(
        "--vocab_size", 
        type=int,
        help="Vocabulary size of the tokenizer",
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
        choices=["ar", "ja", "de", "sw", "th", "hi", "el", "my", "si", "te"]
    )
    parser.add_argument(
        "--num_new_tokens", 
        type=int,
        help="Number of new tokens to add to the tokenizer",
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
    