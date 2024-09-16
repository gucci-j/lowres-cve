import pandas as pd
from pathlib import Path

def main(args):
    # Load the FLORES data for 8 languages
    lang_code_to_flores_code = {
        "en": "eng_Latn",
        "ar": "arb_Arab",
        "de": "deu_Latn",
        "el": "ell_Grek",
        "hi": "hin_Deva",
        "ja": "jpn_Jpan",
        "sw": "swh_Latn",
        "th": "tha_Thai",
        "my": "mya_Mymr",
        "si": "sin_Sinh",
        "te": "tel_Telu",
    }

    # Create the dev dataset
    subset_name = "dev"
    results = []
    for lang_code, flores_code in lang_code_to_flores_code.items():
        file_path = Path(args.data_dir) / subset_name / str(subset_name + "." + flores_code)
        with open(file_path, "r") as f:
            data = f.readlines()
        data = [line.strip() for line in data]
        if results == []:
            results = [{lang_code: line} for line in data]
        else:
            for index, line in enumerate(data):
                results[index][lang_code] = line
    # Convert into DataFrame
    dev_df = pd.DataFrame(results)
    # Save the data as jsonl
    dev_df.to_json(args.output_dir + f"/dev.jsonl", lines=True, orient="records", force_ascii=False)

    # Create the test dataset
    subset_name = "devtest"
    results = []
    for lang_code, flores_code in lang_code_to_flores_code.items():
        file_path = Path(args.data_dir) / subset_name / str(subset_name + "." + flores_code)
        with open(file_path, "r") as f:
            data = f.readlines()
        data = [line.strip() for line in data]
        if results == []:
            results = [{lang_code: line} for line in data]
        else:
            for index, line in enumerate(data):
                results[index][lang_code] = line
    # Convert into DataFrame
    devtest_df = pd.DataFrame(results)
    # Sample 500 examples
    devtest_df = devtest_df.sample(n=500, random_state=42)
    # Save the data as jsonl
    devtest_df.to_json(args.output_dir + f"/test.jsonl", lines=True, orient="records", force_ascii=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create HF datasets')
    parser.add_argument(
        "--data_dir",
        type=str,
        help="The directory where the data is stored",
    )
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
