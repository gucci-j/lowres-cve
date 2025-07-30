from pathlib import Path
import fasttext
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


def main(args):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        cache_dir=args.cache_dir
    )
    
    # Load the dataset
    dataset = load_dataset(
        "text", 
        data_files=args.text_path,
        cache_dir=args.cache_dir,
        split="train",
    )

    # Tokenize the dataset
    dataset = dataset.map(
        lambda sample: {"text": " ".join([token for token in tokenizer.tokenize(sample["text"])])},
        num_proc=args.num_proc,
    )
    cache_path = Path(args.data_dir) / f"tokenized_text_{args.lang_code}.txt"
    with cache_path.open("w+", encoding="utf-8") as f:
        f.writelines((text + "\n" for text in tqdm(dataset["text"], desc="Writing data...")))

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
        thread=args.num_proc // 2,
        minCount=configs["min_count"],
    )

    # Save the FastText model
    model_path = Path(args.output_dir) / f"fasttext_model_{args.model_abbrev}_{args.lang_code}.bin"
    fasttext_model.save_model(str(model_path))

    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Train a fasttext model.")
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        required=True,
        help="The tokenizer name or path."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The cache directory."
    )
    parser.add_argument(
        "--text_path",
        type=str,
        required=True,
        help="The path to the text file."
    )
    parser.add_argument(
        "--lang_code",
        type=str,
        required=True,
        help="The target language."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="The data cache directory."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory."
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=4,
        help="The number of processes to use for tokenization."
    )
    parser.add_argument(
        "--model_abbrev",
        type=str,
        default="",
        help="The abbreviation for the model."
    )
    args = parser.parse_args()
    main(args)