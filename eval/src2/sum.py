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


def lang_code_to_instruction(lang_code: str) -> str:
    """Converts a language code to an instruction to summarize the text in that language.

    Args:
        lang_code: The language code

    Returns:
        The instruction in the specified language.

    Raises:
        ValueError: If the language code is unknown.
    """
    if lang_code == "en":
        return "Summarize the following text in English:"
    elif lang_code == "si":
        return "පහත පාඨයේ සාරාංශය සිංහලෙන් ලියන්න. ලිපිය:"
    elif lang_code == "my":
        return "အောက်ပါစာသားကို မြန်မာဘာသာဖြင့် အကျဉ်းချုပ်ရေးပါ။ ဆောင်းပါး:"
    elif lang_code == "te":
        return "క్రింది వచనం యొక్క సారాంశం తెలుగులో రాయండి. వ్యాసం:"
    else:
        raise ValueError(f"Unknown language code: {lang_code}")


def lang_code_to_anchor(lang_code: str) -> str:
    """Converts a language code to an anchor to summarize the text in that language.

    Args:
        lang_code: The language code

    Returns:
        The anchor in the specified language.

    Raises:
        ValueError: If the language code is unknown.
    """
    if lang_code == "en":
        return "Summary:"
    elif lang_code == "si":
        return "සාරාංශය:"
    elif lang_code == "my":
        return "အကျဉ်းချုပ်:"
    elif lang_code == "te":
        return "సారాంశం:"
    else:
        raise ValueError(f"Unknown language code: {lang_code}")


def buffer_fn(
    instruction: str,
    anchor: str,
):
    def prompt_fn(line, task_name: str):
        summary = line["summary"]
        text = line["text"]
        return Doc(
            task_name=task_name,
            query=f"{instruction} {text} {anchor}",
            gold_index=0,
            choices=[str(summary)],
            specific={"text": text}
        )
    return prompt_fn


for language in [
    "en",
    "my",
    "te",
    "si",
]:
    if language == "en":
        task = LightevalTaskConfig(
            name=f"sum:{language}",
            prompt_function=buffer_fn(
                instruction=lang_code_to_instruction(language),
                anchor=lang_code_to_anchor(language),
            ),
            suite=("custom",),
            hf_repo="your-hub-id/sum-en",
            hf_subset="default",
            evaluation_splits=("test",),
            hf_avail_splits=["test"],
            metric=[chrf_sample],
            generation_size=128,
            stop_sequence=["\n"],
            trust_dataset=True,
        )
    else:
        task = LightevalTaskConfig(
            name=f"sum:{language}",
            prompt_function=buffer_fn(
                instruction=lang_code_to_instruction(language),
                anchor=lang_code_to_anchor(language),
            ),
            suite=("custom",),
            hf_repo=f"your-hub-id/sum-{language}",
            hf_subset="default",
            evaluation_splits=("test",),
            hf_avail_splits=["test"],
            metric=[chrf_sample],
            generation_size=128,
            stop_sequence=["\n"],
            trust_dataset=True,
        )
    TASKS_TABLE.append(task)
