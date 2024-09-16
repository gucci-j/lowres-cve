

import json
import os
import pandas as pd
import datasets

from transformers import AutoTokenizer

lang_code_to_lang_name = {
    "de": "german",
    "el": "greek",
    "ja": "japanese",
    "ar": "arabic",
    "sw": "swahili",
    "th": "thai",
    "hi": "hindi", ,
    "my": "burmese",
    "si": "sinhala",
    "te": "telugu"
}

def restructure_data(data: dict, lang_code: str, tokenizer: AutoTokenizer) -> dict:
    if lang_code == "de":
        return {
            "id": data["gem_id"],
            "url": data["url"],
            "title": data["title"],
            "summary": data["target"],
            "text": data["text"],
            "len": len(tokenizer.encode(data["text"]))
        }
    else:
        return {
                "id": data["id"],
                "url": data["url"],
                "title": data["title"],
                "summary": data["summary"],
                "text": data["text"],
                "len": len(tokenizer.encode(data["text"]))
            }


def main(args):
    if args.lang_code == "de":
        task_name = 'GEM/mlsum'
        subset_name = args.lang_code
    elif args.lang_code == "el":
        task_name = "your-hf-id/GreekSUM"
        subset_name = None
    else:
        task_name = "csebuetnlp/xlsum"
        subset_name = lang_code_to_lang_name[args.lang_code]
    dataset = datasets.load_dataset(
        task_name, subset_name, split="test+validation",
        cache_dir=args.cache_dir
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        cache_dir=args.cache_dir
    )

    # Load the data
    test_data = []
    for index, sample in enumerate(dataset):
        data = restructure_data(sample, lang_code=args.lang_code, tokenizer=tokenizer)
        if data is not None:
            test_data.append(data)
    
    test_data = sorted(test_data, key=lambda x: x["len"])[:500]

    # Convert into DataFrame
    test_df = pd.DataFrame(test_data)

    # Save the data as .csv
    test_df.to_json(args.output_dir + "/test.jsonl", lines=True, orient="records", force_ascii=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create HF datasets')
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="The directory to save the downloaded files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The directory to save the output files",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        help="The name of the repository to which the data will be uploaded",
    )
    parser.add_argument(
        "--lang_code",
        type=str,
        help="The language code of the dataset",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="The path or name of a tokenizer."
    )
    args = parser.parse_args()
    main(args)


    from huggingface_hub import HfApi
    api = HfApi()
    try:
        api.create_repo(
            repo_id=args.repo_id, 
            private=True,
            repo_type='dataset',
        )
    except Exception:
        pass
    api.upload_folder(
        folder_path=args.output_dir,
        repo_id=args.repo_id,
        repo_type='dataset',
    )
