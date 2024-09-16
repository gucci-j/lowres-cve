"""
SUM evaluation tasks for lighteval.
"""

import numpy as np
from aenum import extend_enum
from rouge import Rouge
from rouge_score import rouge_scorer
from sumeval.metrics.rouge import RougeCalculator
from transformers import AutoTokenizer
import evaluate

from lighteval.metrics import Metrics
from lighteval.metrics.metrics import SampleLevelMetric
from lighteval.metrics.metrics_sample import ROUGE
from lighteval.metrics.utils import MetricCategory, MetricUseCase
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

task_ja = LightevalTaskConfig(
    name="sum:ja",
    prompt_function="sum_prompt_fn",  
    hf_repo="your-hub-id/sum-ja", # TODO: Need to change here
    hf_subset="default",
    metric=["rougeL_mt5"],
    hf_avail_splits=["test"], 
    evaluation_splits=["test"], 
    few_shots_split=None, 
    few_shots_select=None,
    suite=["custom"],
    generation_size=128,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
)

task_de = LightevalTaskConfig(
    name="sum:de",
    prompt_function="sum_prompt_fn",  
    hf_repo="your-hub-id/sum-de", # TODO: Need to change here
    hf_subset="default",
    metric=["rougeL_mt5"], 
    hf_avail_splits=["test"], 
    evaluation_splits=["test"], 
    few_shots_split=None, 
    few_shots_select=None,
    suite=["custom"],
    generation_size=128,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
)

task_ar = LightevalTaskConfig(
    name="sum:ar",
    prompt_function="sum_prompt_fn",  
    hf_repo="your-hub-id/sum-ar", # TODO: Need to change here
    hf_subset="default",
    metric=["rougeL_mt5"], 
    hf_avail_splits=["test"], 
    evaluation_splits=["test"], 
    few_shots_split=None, 
    few_shots_select=None,
    suite=["custom"],
    generation_size=128,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
)

task_sw = LightevalTaskConfig(
    name="sum:sw",
    prompt_function="sum_prompt_fn",  
    hf_repo="your-hub-id/sum-sw", # TODO: Need to change here
    hf_subset="default",
    metric=["rougeL_mt5"], 
    hf_avail_splits=["test"], 
    evaluation_splits=["test"], 
    few_shots_split=None, 
    few_shots_select=None,
    suite=["custom"],
    generation_size=128,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
)

task_th = LightevalTaskConfig(
    name="sum:th",
    prompt_function="sum_prompt_fn",  
    hf_repo="your-hub-id/sum-th", # TODO: Need to change here
    hf_subset="default",
    metric=["rougeL_mt5"], 
    hf_avail_splits=["test"], 
    evaluation_splits=["test"], 
    few_shots_split=None, 
    few_shots_select=None,
    suite=["custom"],
    generation_size=128,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
)

task_hi = LightevalTaskConfig(
    name="sum:hi",
    prompt_function="sum_prompt_fn",  
    hf_repo="your-hub-id/sum-hi", # TODO: Need to change here
    hf_subset="default",
    metric=["rougeL_mt5"], 
    hf_avail_splits=["test"], 
    evaluation_splits=["test"], 
    few_shots_split=None, 
    few_shots_select=None,
    suite=["custom"],
    generation_size=128,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
)

task_el = LightevalTaskConfig(
    name="sum:el",
    prompt_function="sum_prompt_fn",  
    hf_repo="your-hub-id/sum-el", # TODO: Need to change here
    hf_subset="default",
    metric=["rougeL_mt5"], 
    hf_avail_splits=["test"], 
    evaluation_splits=["test"], 
    few_shots_split=None, 
    few_shots_select=None,
    suite=["custom"],
    generation_size=128,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
)

task_my = LightevalTaskConfig(
    name="sum:my",
    prompt_function="sum_prompt_fn",  
    hf_repo="your-hub-id/sum-my", # TODO: Need to change here
    hf_subset="default",
    metric=["rougeL_mt5"], 
    hf_avail_splits=["test"], 
    evaluation_splits=["test"], 
    few_shots_split=None, 
    few_shots_select=None,
    suite=["custom"],
    generation_size=128,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
)

task_si = LightevalTaskConfig(
    name="sum:si",
    prompt_function="sum_prompt_fn",  
    hf_repo="your-hub-id/sum-si", # TODO: Need to change here
    hf_subset="default",
    metric=["rougeL_mt5"], 
    hf_avail_splits=["test"], 
    evaluation_splits=["test"], 
    few_shots_split=None, 
    few_shots_select=None,
    suite=["custom"],
    generation_size=128,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
)

task_te = LightevalTaskConfig(
    name="sum:te",
    prompt_function="sum_prompt_fn",  
    hf_repo="your-hub-id/sum-te", # TODO: Need to change here
    hf_subset="default",
    metric=["rougeL_mt5"], 
    hf_avail_splits=["test"], 
    evaluation_splits=["test"], 
    few_shots_split=None, 
    few_shots_select=None,
    suite=["custom"],
    generation_size=128,
    stop_sequence=["\n"],
    output_regex=None,
    frozen=False,
)


def sum_prompt_fn(
    line, 
    task_name: str = None
):
    summary = line["summary"]
    text = line["text"]
    lang_code = task_name.split(":")[1]

    if lang_code == "ja":
        return Doc(
            task_name=task_name,
            query=f"次の文章の要約を日本語で書きなさい。記事: {text} 要約:",
            gold_index=0,
            choices=[str(summary)],
            specific={"text": text},
        )
    
    elif lang_code == "de":
        return Doc(
            task_name=task_name,
            query=f"Schreiben Sie eine kurze Zusammenfassung des folgenden Textes auf Deutsch. Artikel: {text} Zusammenfassung:",
            gold_index=0,
            choices=[str(summary)],
            specific={"text": text},
        )

    elif lang_code == "ar":
        return Doc(
            task_name=task_name,
            query=f"اكتب ملخصًا قصيرًا للنص التالي باللغة العربية. المقالة: {text} الملخص:",
            gold_index=0,
            choices=[str(summary)],
            specific={"text": text},
        )

    elif lang_code == "sw":
        return Doc(
            task_name=task_name,
            query=f"Andika muhtasari mfupi wa maandishi yafuatayo kwa Kiswahili. Makala: {text} Muhtasari:",
            gold_index=0,
            choices=[str(summary)],
            specific={"text": text},
        )

    elif lang_code == "th":
        return Doc(
            task_name=task_name,
            query=f"เขียนสรุปสั้น ๆ ของข้อความต่อไปนี้เป็นภาษาไทย บทความ: {text} สรุป:",
            gold_index=0,
            choices=[str(summary)],
            specific={"text": text},
        )

    elif lang_code == "hi":
        return Doc(
            task_name=task_name,
            query=f"निम्नलिखित का संक्षेप हिंदी में लिखे। लेख: {text} संक्षेप:",
            gold_index=0,
            choices=[str(summary)],
            specific={"text": text},
        )

    elif lang_code == "el":
        return Doc(
            task_name=task_name,
            query=f"Γράψε μια σύντομη περίληψη του παρακάτω κειμένου στα ελληνικά. Άρθρο: {text} Περίληψη:",
            gold_index=0,
            choices=[str(summary)],
            specific={"text": text},
        )
    
    elif lang_code == "my":
        return Doc(
            task_name=task_name,
            query=f"အောက်ပါစာသားကို မြန်မာဘာသာဖြင့် အကျဉ်းချုပ်ရေးပါ။ ဆောင်းပါး: {text} အကျဉ်းချုပ်:",
            gold_index=0,
            choices=[str(summary)],
            specific={"text": text},
        )
    
    elif lang_code == "si":
        return Doc(
            task_name=task_name,
            query=f"පහත පාඨයේ සාරාංශය සිංහලෙන් ලියන්න. ලිපිය: {text} සාරාංශය:",
            gold_index=0,
            choices=[str(summary)],
            specific={"text": text},
        )
    
    elif lang_code == "te":
        return Doc(
            task_name=task_name,
            query=f"క్రింది వచనం యొక్క సారాంశం తెలుగులో రాయండి. వ్యాసం: {text} సారాంశం:",
            gold_index=0,
            choices=[str(summary)],
            specific={"text": text},
        )
    
    else:
        raise ValueError(f"Language code {lang_code} is not supported.")


# CUSTOM METRIC IF NEEDED
class ROUGEMT5(ROUGE):
    ALLOWED_ROUGE_METHODS = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def _rouge_score(self, golds: list[str], preds: list[str]):
        scores = {m: [] for m in self.methods}
        for pred in preds:
            _pred = ' '.join(self._tokenizer.tokenize(pred))
            for gold in golds:
                _gold = ' '.join(self._tokenizer.tokenize(gold))
                cur_scores = self.scorer.score(_gold, _pred)
                for method in self.methods:
                    scores[method].append(cur_scores[method].fmeasure)
        return {method: self.aggregation_function(scores[method]) for method in self.methods}

rougeL_mt5 = SampleLevelMetric(
    metric="rougeL",
    sample_level_fn=ROUGEMT5("rougeL").compute,
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.SUMMARIZATION,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)
extend_enum(Metrics, "rougeL_mt5", rougeL_mt5)


# MODULE LOGIC
_TASKS = (
    task_ja,
    task_de,
    task_ar,
    task_sw,
    task_th,
    task_hi,
    task_el,
    task_my,
    task_si,
    task_te,
)
TASKS_TABLE = [task.as_dict() for task in _TASKS]

if __name__ == "__main__":
    print(t["name"] for t in TASKS_TABLE)
    print(len(TASKS_TABLE))
