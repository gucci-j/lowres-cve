import json
from pathlib import Path

import numpy as np
import pandas as pd
from bleurt import score
from tqdm import tqdm


class BLEURT:
    def __init__(self, checkpoint: str):
        self.scorer = score.BleurtScorer(checkpoint)
    
    def compute(self, golds: list[str], predictions: list[str]) -> float:
        """Uses the stored BLEURT scorer to compute the score on the current sample.

        Args:
            golds (list[str]): Reference targets
            predictions (list[str]): Predicted strings

        Returns:
            float: Score over the current sample's items.
        """
        scores = self.scorer.score(references=golds, candidates=predictions)
        assert isinstance(scores, list) and len(scores) == 1
        return scores.pop()


def calc_bleurt(parquet_file_path: Path, scorer: BLEURT) -> float:
    df = pd.read_parquet(parquet_file_path, engine='pyarrow')
    scores = []
    pbar = tqdm(total=len(df))
    for index, row in df.iterrows():
        score = scorer.compute(row.gold, row.predictions)
        scores.append(score)
        pbar.update(1)
    return np.mean(scores), np.std(scores)


def main(args):
    parquet_dir = Path(args.parquet_dir)
    scorer = BLEURT(checkpoint=args.checkpoint_path)

    for file_path in parquet_dir.glob("**/*.parquet"):
        print("Now computing for", file_path)
        # sanity check
        result_file_name = file_path.stem + ".json"
        if file_path.parent.parent.parent.stem == "details":
            # adapted
            time_stamp = file_path.parent.name
            model_name = file_path.parent.parent.name
            result_path = parquet_dir.parent / "results" / model_name / result_file_name
        else:
            # source
            time_stamp = file_path.parent.name
            model_name = file_path.parent.parent.name
            org_name = file_path.parent.parent.parent.stem
            result_path = parquet_dir.parent / "results" / org_name / model_name / result_file_name
        result_rougel_path = result_path.parent / "".join(["results_", time_stamp, ".json"])
        if result_path.exists():
            print("\tSkipping due to the already processed file.")
            continue
        if "bleurt" in result_file_name:
            print("\tSkipping due to the already processed file.")
            continue
        if not result_rougel_path.exists():
            print("\tSkipping due to no source result file found with", result_rougel_path)
            continue
        if args.keyword is not None:
            if args.keyword not in model_name:
                print("\tSkipping due to no keyword match with", args.keyword)
                continue

        print("Results will be saved at", result_path.resolve())
        # calc bleurt
        bleurt_score, bleurt_std = calc_bleurt(file_path, scorer)

        # save
        results = {
            "results": {"bleurt": bleurt_score, "bleurt_stderr": bleurt_std}
        }
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_dir", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--keyword", type=str, default=None)
    args = parser.parse_args()

    main(args)
