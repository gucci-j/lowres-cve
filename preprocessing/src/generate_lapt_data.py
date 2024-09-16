from datasets import load_dataset
from transformers import AutoTokenizer, LlamaTokenizer


def group_texts(examples: dict, block_size=128, eos_id=2):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    else:
        # add padding
        for k, v in concatenated_examples.items():
            concatenated_examples[k] = v + [eos_id] * (block_size - len(v))
        total_length = block_size
    
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()    
    return result


def main(args):
    # Load the dataset
    print("Loading dataset...")
    dataset = load_dataset(
        "text", 
        data_files={"train": args.data_path},
        cache_dir=args.datasets_cache_dir,
        split="train"
    )

    # Load the tokenizer
    print("Loading tokenizer...")
    if args.tokenizer_name_or_path == "mistralai/Mistral-7B-v0.1":
        tokenizer = LlamaTokenizer.from_pretrained(
            args.tokenizer_name_or_path,
            cache_dir=args.tokenizer_cache_dir
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name_or_path,
            cache_dir=args.tokenizer_cache_dir
        )
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize the dataset
    print("Tokenizing the dataset...")
    dataset = dataset.map(
        lambda examples: tokenizer(examples["text"]),
        batched=True,
        num_proc=4,
        remove_columns=dataset.column_names,
    )

    # Group the texts
    print("Grouping the texts...")
    dataset = dataset.map(
        lambda examples: group_texts(examples, args.max_length),
        batched=True, 
        num_proc=4
    )

    # Save the tokenized dataset to a file
    print("Saving tokenized dataset...")
    dataset.save_to_disk(args.output_data_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", 
        type=str, 
        required=True, 
        help="Path to the input data file"
    )
    parser.add_argument(
        "--output_data_path", 
        type=str, 
        required=True, 
        help="Path to the output data file"
    )
    parser.add_argument(
        "--datasets_cache_dir", 
        type=str, 
        default=None,
        help="Directory to cache datasets"
    )
    parser.add_argument(
        "--tokenizer_name_or_path", 
        type=str, 
        required=True,
        help="Name or path of the tokenizer to use"
    )
    parser.add_argument(
        "--tokenizer_cache_dir", 
        type=str, 
        default=None,
        help="Directory to cache tokenizers"
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=4, 
        help="Number of worker processes to use"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=128, 
        help="Maximum length of the tokenized sequences"
    )
    args = parser.parse_args()
    main(args)
