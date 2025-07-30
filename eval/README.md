Evaluation
===

Here, we evaluate models on machine translation and summarization tasks. We use the `lighteval` library to evaluate the models.

## Reproduction
First, you need to download datasets:
* FLORES-200: https://github.com/openlanguagedata/flores
* GreekSUM: As noted in the Appendix, you need to generate a summarization dataset for Greek from scratch by following https://github.com/iakovosevdaimon/GreekSUM.

After that, you just need to run the following scripts:
* Preprocessing the data:
    * For machine translation: [create_mt_dataset.sh](./scripts/preprocessing/create_mt_dataset.sh)
    * For summarization: [create_sum_dataset.sh](./scripts/preprocessing/create_sum_dataset.sh) 
* Llama2: [llama2.sh](./scripts/llama2.sh)
* Llama3: [llama3.sh](./scripts/llama3.sh)
* Gemma2: [gemma2.sh](./scripts/gemma2.sh)

Note that you need to modify the paths starting with `/path/to/` and `your-hub-id` in each script and python scripts under `src/`. Also, you need to change the dataset ids in the python scripts (`mt.py` and `sum.py`).

### Classification tasks and English tasks
For classification tasks and English tasks, please use scripts under `lowres-cve/eval/scripts2/`. The scripts are similar to the ones in `lowres-cve/eval/scripts/`, but they use more recent versions of the `lighteval` library and have been updated to work with the latest models.

* [Llama2]: [llama2.sh](./scripts2/llama2.sh)
* [Llama3]: [llama3.sh](./scripts2/llama3.sh)
* [Gemma2]: [gemma2.sh](./scripts2/gemma2.sh)

We also include scripts for summarization and machine translation tasks using the more recent `lighteval` library for your convenience.


## Reference
* https://github.com/huggingface/lighteval
