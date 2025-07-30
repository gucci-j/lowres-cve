from aenum import extend_enum
import numpy as np

from lighteval.metrics.metrics import Metrics, SampleLevelMetric
from lighteval.metrics.utils.metric_utils import MetricCategory, MetricUseCase
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


TASKS_TABLE = []


# CUSTOM METRIC IF NEEDED
class SampleLevelTranslationMetric:
    def __init__(self, metric_type: str):
        """Stores the relevant parameters for a corpus level translation metric.

        Args:
            metric_type (str): Can be any of bleu, chrf, or ter depending on the metric to use.
        """
        import sacrebleu
        self.metric_type = metric_type
        if metric_type == "bleu":
            self.metric = sacrebleu.sentence_bleu
        elif metric_type == "chrf":
            self.metric = sacrebleu.sentence_chrf
        elif metric_type == "chrf++":
            self.metric = sacrebleu.sentence_chrf
        elif metric_type == "ter":
            self.metric = sacrebleu.sentence_ter
        else:
            raise ValueError(f"Unknown corpus level translation metric type : {metric_type}")

    def compute(self, golds: list[str], predictions: list[str], **kwargs) -> float:
        assert len(golds) == 1 and len(predictions) == 1
        if self.metric_type == "chrf++":
            return float(self.metric(predictions.pop(), golds, word_order=2).score)
        else:
            return float(self.metric(predictions.pop(), golds).score)

chrf_sample = SampleLevelMetric(
    metric_name="chrf_sample",
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.TRANSLATION,
    sample_level_fn=SampleLevelTranslationMetric("chrf").compute, # how to compute score for one sample
    corpus_level_fn=np.mean, # aggregation
    higher_is_better=True,
)
extend_enum(Metrics, "chrf_sample", chrf_sample)


def lang_code_to_2tgt_instruction(lang_code: str) -> str:
    """Converts a language code to an instruction to translate from English to target.

    Args:
        lang_code: The language code 

    Returns:
        The translation instruction string.

    Raises:
        ValueError: If the language code is unknown.
    """
    if lang_code == "my":
        # Burmese
        return "အင်္ဂလိပ်မှ မြန်မာသို့ ဘာသာပြန်ပါ။:\n"
    elif lang_code == "si":
        # Sinhala
        return "ඉංග්‍රීසි සිංහලයට පරිවර්තනය කරන්න:\n"
    elif lang_code == "te":
        # Telugu
        return "ఆంగ్లం నుండి తెలుగుకు అనువదించండి:\n"
    else:
        raise ValueError(f"Unknown language code: {lang_code}")


def buffer_fn_2tgt(
    language: str, 
    instruction: str,
):
    def prompt_fn(line, task_name: str):
        return Doc(
            task_name=task_name,
            query=f"{instruction}{line['en']} =",
            gold_index=0,
            choices=[line[language]],
            instruction=instruction,
        )
    return prompt_fn


for language in [
    "my",  # Burmese
    "si",  # Sinhala
    "te",  # Telugu
]:
    task = LightevalTaskConfig(
        name=f"mt:en2{language}",
        prompt_function=buffer_fn_2tgt(
            language=language,
            instruction=lang_code_to_2tgt_instruction(language),
        ),
        suite=("custom",),
        hf_repo="your-hub-id/flores", # TODO: Need to change here
        hf_subset="default",
        evaluation_splits=("test",),
        hf_avail_splits=["validation", "test"],
        metric=[chrf_sample],
        generation_size=128,
        stop_sequence=["\n"],
        trust_dataset=True,
    )
    TASKS_TABLE.append(task)
