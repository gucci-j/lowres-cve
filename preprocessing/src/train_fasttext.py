from pathlib import Path

import fasttext
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def filter_fn(example, min_length: int = 5, target_lang: str = "sw"):
    if target_lang not in ("ja", "th"):
        if min_length <= len(example["text"].split(' ')):
            return True
        return False
    else:
        if min_length <= len(example["text"]):
            return True
        return False


def main(args):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        cache_dir=args.tokenizer_cache_dir
    )
    
    if not args.restart_from_cache:
        # Load the dataset
        dataset = load_dataset(
            "text", 
            data_files=args.text_path,
            cache_dir=args.data_cache_dir,
            split="train",
        )
        print("Before filtering:", dataset)

        # Filter the dataset
        dataset = dataset.filter(
            lambda example: filter_fn(
                example,
                args.min_length,
                args.target_lang
            )
        )
        print("After filtering:", dataset)

        # Tokenize the dataset
        dataset = dataset.map(
            lambda sample: {"text": " ".join([token for token in tokenizer.tokenize(sample["text"])])},
            num_proc=4,
        )

        # Save the tokenized dataset
        cache_path = Path(args.data_cache_dir) / f"tokenized_text_{args.target_lang}_30K.txt"
        with cache_path.open("w+", encoding="utf-8") as f:
            f.writelines((text + "\n" for text in tqdm(dataset["text"], desc="Writing data...")))
    else:
        cache_path = Path(args.data_cache_dir) / f"tokenized_text_{args.target_lang}_30K.txt"

    # Train the FastText model
    configs = {
        "dim": 300,
        "epochs": 3,
        "min_count": 10,
    }
    fasttext_model = fasttext.train_unsupervised(
        str(cache_path),
        dim=configs["dim"],
        neg=10,
        model="cbow",
        epoch=configs["epochs"],
        thread=4,
        minCount=configs["min_count"],
    )

    # Save the FastText model
    model_path = Path(args.model_cache_dir) / f"fasttext_model_{args.target_lang}_30K.bin"
    fasttext_model.save_model(str(model_path))

    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Train a fasttext model with the CC-100 corpus.")
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        required=True,
        help="The tokenizer name or path."
    )
    parser.add_argument(
        "--tokenizer_cache_dir",
        type=str,
        default=None,
        help="The tokenizer cache directory."
    )
    parser.add_argument(
        "--text_path",
        type=str,
        required=True,
        help="The path to the text file."
    )
    parser.add_argument(
        "--data_cache_dir",
        type=str,
        default=None,
        help="The cache directory."
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=5,
        help="The minimum length."
    )
    parser.add_argument(
        "--target_lang",
        type=str,
        required=True,
        choices=["ja", "de", "sw", "ar", "th", "hi", "el", "te", "si", "my"],
        help="The target language."
    )
    parser.add_argument(
        "--model_cache_dir",
        type=str,
        required=True,
        help="The model cache directory."
    )
    parser.add_argument(
        "--restart_from_cache",
        action="store_true",
        help="Whether to restart from the cache."
    )

    args = parser.parse_args()

    main(args)
