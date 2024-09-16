
"""
MT evaluation tasks for lighteval.
"""

import numpy as np
from aenum import extend_enum

from lighteval.metrics import Metrics
from lighteval.metrics.metrics import SampleLevelMetric
from lighteval.metrics.utils import MetricCategory, MetricUseCase
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


task_ja2en = LightevalTaskConfig(
    name="mt:ja2en",
    prompt_function="mt_prompt_fn",  
    hf_repo="your-hub-id/flores", # TODO: Need to change here
    hf_subset="default",
    metric=["chrf_sample"],
    hf_avail_splits=["test", "validation"], 
    evaluation_splits=["test"], 
    few_shots_split=["validation"], 
    few_shots_select="random_sampling_from_train",
    suite=["custom"],
    generation_size=128,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
    trust_dataset=True
)

task_en2ja = LightevalTaskConfig(
    name="mt:en2ja",
    prompt_function="mt_prompt_fn",  
    hf_repo="your-hub-id/flores", # TODO: Need to change here
    hf_subset="default",
    metric=["chrf_sample"],
    hf_avail_splits=["test", "validation"], 
    evaluation_splits=["test"], 
    few_shots_split=["validation"], 
    few_shots_select="random_sampling_from_train",
    suite=["custom"],
    generation_size=128,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
    trust_dataset=True
)

task_ar2en = LightevalTaskConfig(
    name="mt:ar2en",
    prompt_function="mt_prompt_fn",  
    hf_repo="your-hub-id/flores", # TODO: Need to change here
    hf_subset="default",
    metric=["chrf_sample"],
    hf_avail_splits=["test", "validation"], 
    evaluation_splits=["test"], 
    few_shots_split=["validation"], 
    few_shots_select="random_sampling_from_train",
    suite=["custom"],
    generation_size=128,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
    trust_dataset=True
)

task_en2ar = LightevalTaskConfig(
    name="mt:en2ar",
    prompt_function="mt_prompt_fn",  
    hf_repo="your-hub-id/flores", # TODO: Need to change here
    hf_subset="default",
    metric=["chrf_sample"],
    hf_avail_splits=["test", "validation"], 
    evaluation_splits=["test"], 
    few_shots_split=["validation"], 
    few_shots_select="random_sampling_from_train",
    suite=["custom"],
    generation_size=128,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
    trust_dataset=True
)

task_de2en = LightevalTaskConfig(
    name="mt:de2en",
    prompt_function="mt_prompt_fn",  
    hf_repo="your-hub-id/flores", # TODO: Need to change here
    hf_subset="default",
    metric=["chrf_sample"],
    hf_avail_splits=["test", "validation"], 
    evaluation_splits=["test"], 
    few_shots_split=["validation"], 
    few_shots_select="random_sampling_from_train",
    suite=["custom"],
    generation_size=128,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
    trust_dataset=True
)

task_en2de = LightevalTaskConfig(
    name="mt:en2de",
    prompt_function="mt_prompt_fn",  
    hf_repo="your-hub-id/flores", # TODO: Need to change here
    hf_subset="default",
    metric=["chrf_sample"],
    hf_avail_splits=["test", "validation"], 
    evaluation_splits=["test"], 
    few_shots_split=["validation"], 
    few_shots_select="random_sampling_from_train",
    suite=["custom"],
    generation_size=128,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
    trust_dataset=True
)

task_el2en = LightevalTaskConfig(
    name="mt:el2en",
    prompt_function="mt_prompt_fn",  
    hf_repo="your-hub-id/flores", # TODO: Need to change here
    hf_subset="default",
    metric=["chrf_sample"],
    hf_avail_splits=["test", "validation"], 
    evaluation_splits=["test"], 
    few_shots_split=["validation"], 
    few_shots_select="random_sampling_from_train",
    suite=["custom"],
    generation_size=128,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
    trust_dataset=True
)

task_en2el = LightevalTaskConfig(
    name="mt:en2el",
    prompt_function="mt_prompt_fn",  
    hf_repo="your-hub-id/flores", # TODO: Need to change here
    hf_subset="default",
    metric=["chrf_sample"],
    hf_avail_splits=["test", "validation"], 
    evaluation_splits=["test"], 
    few_shots_split=["validation"], 
    few_shots_select="random_sampling_from_train",
    suite=["custom"],
    generation_size=128,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
    trust_dataset=True
)

task_hi2en = LightevalTaskConfig(
    name="mt:hi2en",
    prompt_function="mt_prompt_fn",  
    hf_repo="your-hub-id/flores", # TODO: Need to change here
    hf_subset="default",
    metric=["chrf_sample"],
    hf_avail_splits=["test", "validation"], 
    evaluation_splits=["test"], 
    few_shots_split=["validation"], 
    few_shots_select="random_sampling_from_train",
    suite=["custom"],
    generation_size=128,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
    trust_dataset=True
)

task_en2hi = LightevalTaskConfig(
    name="mt:en2hi",
    prompt_function="mt_prompt_fn",  
    hf_repo="your-hub-id/flores", # TODO: Need to change here
    hf_subset="default",
    metric=["chrf_sample"],
    hf_avail_splits=["test", "validation"], 
    evaluation_splits=["test"], 
    few_shots_split=["validation"], 
    few_shots_select="random_sampling_from_train",
    suite=["custom"],
    generation_size=128,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
    trust_dataset=True
)

task_sw2en = LightevalTaskConfig(
    name="mt:sw2en",
    prompt_function="mt_prompt_fn",  
    hf_repo="your-hub-id/flores", # TODO: Need to change here
    hf_subset="default",
    metric=["chrf_sample"],
    hf_avail_splits=["test", "validation"], 
    evaluation_splits=["test"], 
    few_shots_split=["validation"], 
    few_shots_select="random_sampling_from_train",
    suite=["custom"],
    generation_size=128,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
    trust_dataset=True
)

task_en2sw = LightevalTaskConfig(
    name="mt:en2sw",
    prompt_function="mt_prompt_fn",  
    hf_repo="your-hub-id/flores", # TODO: Need to change here
    hf_subset="default",
    metric=["chrf_sample"],
    hf_avail_splits=["test", "validation"], 
    evaluation_splits=["test"], 
    few_shots_split=["validation"], 
    few_shots_select="random_sampling_from_train",
    suite=["custom"],
    generation_size=128,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
    trust_dataset=True
)

task_th2en = LightevalTaskConfig(
    name="mt:th2en",
    prompt_function="mt_prompt_fn",  
    hf_repo="your-hub-id/flores", # TODO: Need to change here
    hf_subset="default",
    metric=["chrf_sample"],
    hf_avail_splits=["test", "validation"], 
    evaluation_splits=["test"], 
    few_shots_split=["validation"], 
    few_shots_select="random_sampling_from_train",
    suite=["custom"],
    generation_size=128,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
    trust_dataset=True
)

task_en2th = LightevalTaskConfig(
    name="mt:en2th",
    prompt_function="mt_prompt_fn",  
    hf_repo="your-hub-id/flores", # TODO: Need to change here
    hf_subset="default",
    metric=["chrf_sample"],
    hf_avail_splits=["test", "validation"], 
    evaluation_splits=["test"], 
    few_shots_split=["validation"], 
    few_shots_select="random_sampling_from_train",
    suite=["custom"],
    generation_size=128,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
    trust_dataset=True
)

task_en2my = LightevalTaskConfig(
    name="mt:en2my",
    prompt_function="mt_prompt_fn",  
    hf_repo="your-hub-id/flores", # TODO: Need to change here
    hf_subset="default",
    metric=["chrf_sample"],
    hf_avail_splits=["test", "validation"], 
    evaluation_splits=["test"], 
    few_shots_split=["validation"], 
    few_shots_select="random_sampling_from_train",
    suite=["custom"],
    generation_size=128,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
    trust_dataset=True
)

task_en2si = LightevalTaskConfig(
    name="mt:en2si",
    prompt_function="mt_prompt_fn",  
    hf_repo="your-hub-id/flores", # TODO: Need to change here
    hf_subset="default",
    metric=["chrf_sample"],
    hf_avail_splits=["test", "validation"], 
    evaluation_splits=["test"], 
    few_shots_split=["validation"], 
    few_shots_select="random_sampling_from_train",
    suite=["custom"],
    generation_size=128,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
    trust_dataset=True
)

task_en2te = LightevalTaskConfig(
    name="mt:en2te",
    prompt_function="mt_prompt_fn",  
    hf_repo="your-hub-id/flores", # TODO: Need to change here
    hf_subset="default",
    metric=["chrf_sample"],
    hf_avail_splits=["test", "validation"], 
    evaluation_splits=["test"], 
    few_shots_split=["validation"], 
    few_shots_select="random_sampling_from_train",
    suite=["custom"],
    generation_size=128,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
    trust_dataset=True
)


def mt_prompt_fn(
    line, 
    task_name: str = None
):
    task_name = task_name.split("|")[1]
    if task_name == "mt:ja2en":
        return Doc(
            task_name=task_name,
            query=f"日本語から英語へ翻訳しなさい:\n{line['ja']} =",
            gold_index=0,
            choices=[line["en"]],
            instruction="日本語から英語へ翻訳しなさい:\n",
        )
    elif task_name == "mt:en2ja":
        return Doc(
            task_name=task_name,
            query=f"英語から日本語へ翻訳しなさい:\n{line['en']} =",
            gold_index=0,
            choices=[line["ja"]],
            instruction="英語から日本語へ翻訳しなさい:\n",
        )
    elif task_name == "mt:ar2en":
        return Doc(
            task_name=task_name,
            query=f"ترجم العربية إلى الإنجليزية:\n{line['ar']} =",
            gold_index=0,
            choices=[line["en"]],
            instruction="ترجم العربية إلى الإنجليزية:\n",
        )
    elif task_name == "mt:en2ar":
        return Doc(
            task_name=task_name,
            query=f"ترجم الإنجليزية إلى العربية:\n{line['en']} =",
            gold_index=0,
            choices=[line["ar"]],
            instruction="ترجم الإنجليزية إلى العربية:\n",
        )
    elif task_name == "mt:de2en":
        return Doc(
            task_name=task_name,
            query=f"Übersetzen Sie Deutsch ins Englische:\n{line['de']} =",
            gold_index=0,
            choices=[line["en"]],
            instruction="Übersetzen Sie Deutsch ins Englische:\n",
        )
    elif task_name == "mt:en2de":
        return Doc(
            task_name=task_name,
            query=f"Übersetzen Sie Englisch ins Deutsche:\n{line['en']} =",
            gold_index=0,
            choices=[line["de"]],
            instruction="Übersetzen Sie Englisch ins Deutsche:\n",
        )
    elif task_name == "mt:el2en":
        return Doc(
            task_name=task_name,
            query=f"Μεταφράστε τα ελληνικά στα αγγλικά:\n{line['el']} =",
            gold_index=0,
            choices=[line["en"]],
            instruction="Μεταφράστε τα ελληνικά στα αγγλικά:\n",
        )
    elif task_name == "mt:en2el":
        return Doc(
            task_name=task_name,
            query=f"Μεταφράστε τα αγγλικά στα ελληνικά:\n{line['en']} =",
            gold_index=0,
            choices=[line["el"]],
            instruction="Μεταφράστε τα αγγλικά στα ελληνικά:\n",
        )
    elif task_name == "mt:hi2en":
        return Doc(
            task_name=task_name,
            query=f"हिंदी से अंग्रेजी में अनुवाद करें:\n{line['hi']} =",
            gold_index=0,
            choices=[line["en"]],
            instruction="हिंदी से अंग्रेजी में अनुवाद करें:\n",
        )
    elif task_name == "mt:en2hi":
        return Doc(
            task_name=task_name,
            query=f"अंग्रेजी से हिंदी में अनुवाद करें:\n{line['en']} =",
            gold_index=0,
            choices=[line["hi"]],
            instruction="अंग्रेजी से हिंदी में अनुवाद करें:\n",
        )
    elif task_name == "mt:sw2en":
        return Doc(
            task_name=task_name,
            query=f"Tafsiri Kiswahili hadi Kiingereza:\n{line['sw']} =",
            gold_index=0,
            choices=[line["en"]],
            instruction="Tafsiri Kiswahili hadi Kiingereza:\n",
        )
    elif task_name == "mt:en2sw":
        return Doc(
            task_name=task_name,
            query=f"Tafsiri Kiingereza hadi Kiswahili:\n{line['en']} =",
            gold_index=0,
            choices=[line["sw"]],
            instruction="Tafsiri Kiingereza hadi Kiswahili:\n",
        )
    elif task_name == "mt:th2en":
        return Doc(
            task_name=task_name,
            query=f"แปลภาษาไทยเป็นอังกฤษ:\n{line['th']} =",
            gold_index=0,
            choices=[line["en"]],
            instruction="แปลภาษาไทยเป็นอังกฤษ:\n",
        )
    elif task_name == "mt:en2th":
        return Doc(
            task_name=task_name,
            query=f"แปลภาษาอังกฤษเป็นไทย:\n{line['en']} =",
            gold_index=0,
            choices=[line["th"]],
            instruction="แปลภาษาอังกฤษเป็นไทย:\n",
        )
    elif task_name == "mt:en2my":
        return Doc(
            task_name=task_name,
            query=f"အင်္ဂလိပ်မှ မြန်မာသို့ ဘာသာပြန်ပါ။:\n{line['en']} =",
            gold_index=0,
            choices=[line["my"]],
            instruction="အင်္ဂလိပ်မှ မြန်မာသို့ ဘာသာပြန်ပါ။:\n",
        )
    elif task_name == "mt:en2si":
        return Doc(
            task_name=task_name,
            query=f"ඉංග්‍රීසි සිංහලයට පරිවර්තනය කරන්න:\n{line['en']} =",
            gold_index=0,
            choices=[line["si"]],
            instruction="ඉංග්‍රීසි සිංහලයට පරිවර්තනය කරන්න:\n",
        )
    elif task_name == "mt:en2te":
        return Doc(
            task_name=task_name,
            query=f"ఆంగ్లం నుండి తెలుగుకు అనువదించండి:\n{line['en']} =",
            gold_index=0,
            choices=[line["te"]],
            instruction="ఆంగ్లం నుండి తెలుగుకు అనువదించండి:\n",
        )
    else:
        raise ValueError(f"Task {task_name} is not supported.")


class SampleLevelTranslationMetric:
    def __init__(self, metric_type: str):
        """Stores the relevant parameters for a corpus level translation metric.

        Args:
            metric_type (str): Can be any of bleu, chrf, or ter depending on the metric to use.
        """
        import sacrebleu
        if metric_type == "bleu":
            self.metric = sacrebleu.sentence_bleu
        elif metric_type == "chrf":
            self.metric = sacrebleu.sentence_chrf
        elif metric_type == "ter":
            self.metric = sacrebleu.sentence_ter
        else:
            raise ValueError(f"Unknown corpus level translation metric type : {metric_type}")

    def compute(self, golds: list[str], predictions: list[str], **kwargs) -> float:
        assert len(golds) == 1 and len(predictions) == 1
        return float(self.metric(predictions.pop(), golds).score)

chrf_sample = SampleLevelMetric(
    metric="chrf_sample",
    sample_level_fn=SampleLevelTranslationMetric("chrf").compute,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.TRANSLATION,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)
extend_enum(Metrics, "chrf_sample", chrf_sample)



# MODULE LOGIC
_TASKS = (
    task_ja2en,
    task_en2ja,
    task_ar2en,
    task_en2ar,
    task_de2en,
    task_en2de,
    task_el2en,
    task_en2el,
    task_hi2en,
    task_en2hi,
    task_sw2en,
    task_en2sw,
    task_th2en,
    task_en2th,
    task_en2my,
    task_en2si,
    task_en2te,
)
TASKS_TABLE = [task.as_dict() for task in _TASKS]

if __name__ == "__main__":
    print(t["name"] for t in TASKS_TABLE)
    print(len(TASKS_TABLE))
