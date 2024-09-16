How Can We Effectively Expand the Vocabulary of LLMs with 0.01GB of Target Language Text?
===

This is the official code for the paper titled "[How Can We Effectively Expand the Vocabulary of LLMs with 0.01GB of Target Language Text?](https://arxiv.org/abs/2406.11477)" 

For reproduction, please refer to [Reproduction](#reproduction).  
For models developed in the paper, please refer to [Adapted Models](#adapted-models).

## Requirements
* Python 3.12.4 or later
* CUDA 12.4
* torch
* transformers
* peft
* datasets
* evaluate
* bitsandbytes
* scikit-learn
* sentencepiece
* huggingface-hub
* lighteval
* openai
* tqdm
* pyarrow
* entmax
* fastdist
* rouge-score
* numba
* lighteval
* openai
* tiktoken
* BLEURT==0.0.2 (See below)
* fasttext==0.9.2 (See below)


## Installation
After manually installing `PyTorch` and `transformers`, please run the following.
```bash
# fastText
pip install -r requirements.txt
git clone https://github.com/facebookresearch/fastText.git
cd fastText
pip install .

# BLEURT
git clone https://github.com/google-research/bleurt.git
cd bleurt
pip install .
wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip .
unzip BLEURT-20.zip
```

## Reproduction
### 1. Preprocessing
Please see the `preprocessing` directory for preprocessing.

### 2. Target model initialization
Please see the `initialization` directory for target model initialization.

### 3. Language adaptive pre-training
Please see the `lapt` directory for language adaptive pre-training.

### 4. Evaluation
Please see the `eval` directory for evaluation.


## Adapted Models
All adapted models (168 models for Llama2-7B and 48 models each for Llama3-8B and Gemma2-9B) are available on the Hugging Face Model Hub. If you would like to use these models for practical use, we highly receommend using models adapted with **Align + 2x2 LS + MTP + 512**. Other models are not recommended for practical use and they are provided for analysis purposes only. Please see the discussions and recommendations in the paper.


### Llama2

#### Models used for target vocabulary initialization method analysis
*Not recommended for practical use*
| Approach | Link |
| --- | --- |
| LAPT | [ar](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-ar-30K-lapt) / [my](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-my-30K-lapt) / [de](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-de-30K-lapt) / [el](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-el-30K-lapt) / [hi](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-hi-30K-lapt) / [ja](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-ja-30K-lapt) / [si](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-si-30K-lapt) / [sw](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-sw-30K-lapt) / [te](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-te-30K-lapt) / [th](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-th-30K-lapt) |
| Random | [ar](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-ar-30K-rand) / [my](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-my-30K-rand) / [de](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-de-30K-rand) / [el](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-el-30K-rand) / [hi](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-hi-30K-rand) / [ja](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-ja-30K-rand) / [si](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-si-30K-rand) / [sw](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-sw-30K-rand) / [te](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-te-30K-rand) / [th](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-th-30K-rand) |
| FOCUS | [ar](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-ar-30K-focus) / [my](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-my-30K-focus) / [de](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-de-30K-focus) / [el](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-el-30K-focus) / [hi](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-hi-30K-focus) / [ja](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-ja-30K-focus) / [si](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-si-30K-focus) / [sw](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-sw-30K-focus) / [te](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-te-30K-focus) / [th](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-th-30K-focus) |
| Mean | [ar](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-ar-30K-mean) / [my](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-my-30K-mean) / [de](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-de-30K-mean) / [el](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-el-30K-mean) / [hi](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-hi-30K-mean) / [ja](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-ja-30K-mean) / [si](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-si-30K-mean) / [sw](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-sw-30K-mean) / [te](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-te-30K-mean) / [th](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-th-30K-mean) |
| Merge | [ar](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-ar-30K-merge) / [my](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-my-30K-merge) / [de](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-de-30K-merge) / [el](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-el-30K-merge) / [hi](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-hi-30K-merge) / [ja](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-ja-30K-merge) / [si](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-si-30K-merge) / [sw](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-sw-30K-merge) / [te](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-te-30K-merge) / [th](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-th-30K-merge) |
| Align | [ar](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-ar-30K-align) / [my](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-my-30K-align) / [de](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-de-30K-align) / [el](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-el-30K-align) / [hi](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-hi-30K-align) / [ja](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-ja-30K-align) / [si](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-si-30K-align) / [sw](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-sw-30K-align) / [te](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-te-30K-align) / [th](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-th-30K-align) |


#### Models used for training strategy analysis
*Not recommended for practical use except for 2x2 LS + MTP + 512 models*
|Approach  | Link |
| --- | --- |
| **LoRA** | |
| CLM + 2048 | [ar](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-ar-30K-align) / [my](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-my-30K-align) / [el](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-el-30K-align) / [hi](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-hi-30K-align) / [si](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-si-30K-align) / [te](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-te-30K-align) |
| MTP + 2048 | [ar](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-ar-30K-align-mtp) / [my](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-my-30K-align-mtp) / [el](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-el-30K-align-mtp) / [hi](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-hi-30K-align-mtp) / [si](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-si-30K-align-mtp) / [te](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-te-30K-align-mtp) |
| CLM + 512 | [ar](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-ar-30K-align-512) / [my](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-my-30K-align-512) / [el](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-el-30K-align-512) / [hi](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-hi-30K-align-512) / [si](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-si-30K-align-512) / [te](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-te-30K-align-512) |
| MTP + 512 | [ar](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-ar-30K-align-mtp-512) / [my](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-my-30K-align-mtp-512) / [el](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-el-30K-align-mtp-512) / [hi](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-hi-30K-align-mtp-512) / [si](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-si-30K-align-mtp-512) / [te](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-te-30K-align-mtp-512) |
| **2 stage** | |
| CLM + 2048 | [ar](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-ar-30K-align-2stage) / [my](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-my-30K-align-2stage) / [el](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-el-30K-align-2stage) / [hi](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-hi-30K-align-2stage) / [si](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-si-30K-align-2stage) / [te](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-te-30K-align-2stage) |
| MTP + 2048 | [ar](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-ar-30K-align-2stage-mtp) / [my](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-my-30K-align-2stage-mtp) / [el](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-el-30K-align-2stage-mtp) / [hi](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-hi-30K-align-2stage-mtp) / [si](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-si-30K-align-2stage-mtp) / [te](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-te-30K-align-2stage-mtp) |
| CLM + 512 | [ar](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-ar-30K-align-2stage-512) / [my](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-my-30K-align-2stage-512) / [el](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-el-30K-align-2stage-512) / [hi](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-hi-30K-align-2stage-512) / [si](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-si-30K-align-2stage-512) / [te](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-te-30K-align-2stage-512) |
| MTP + 512 | [ar](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-ar-30K-align-2stage-mtp-512) / [my](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-my-30K-align-2stage-mtp-512) / [el](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-el-30K-align-2stage-mtp-512) / [hi](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-hi-30K-align-2stage-mtp-512) / [si](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-si-30K-align-2stage-mtp-512) / [te](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-te-30K-align-2stage-mtp-512) |
| **2x2 LS** | |
| CLM + 2048 | [ar](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-ar-30K-align-2x2ls) / [my](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-my-30K-align-2x2ls) / [el](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-el-30K-align-2x2ls) / [hi](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-hi-30K-align-2x2ls) / [si](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-si-30K-align-2x2ls) / [te](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-te-30K-align-2x2ls) |
| MTP + 2048 | [ar](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-ar-30K-align-2x2ls-mtp) / [my](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-my-30K-align-2x2ls-mtp) / [el](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-el-30K-align-2x2ls-mtp) / [hi](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-hi-30K-align-2x2ls-mtp) / [si](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-si-30K-align-2x2ls-mtp) / [te](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-te-30K-align-2x2ls-mtp) |
| CLM + 512 | [ar](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-ar-30K-align-2x2ls-512) / [my](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-my-30K-align-2x2ls-512) / [el](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-el-30K-align-2x2ls-512) / [hi](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-hi-30K-align-2x2ls-512) / [si](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-si-30K-align-2x2ls-512) / [te](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-te-30K-align-2x2ls-512) |
| MTP + 512 | [ar](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-ar-30K-align-2x2ls-mtp-512) / [my](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-my-30K-align-2x2ls-mtp-512) / [el](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-el-30K-align-2x2ls-mtp-512) / [hi](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-hi-30K-align-2x2ls-mtp-512) / [si](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-si-30K-align-2x2ls-mtp-512) / [te](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-te-30K-align-2x2ls-mtp-512) |


#### Models used for target vocabulary size analysis
*Not recommended for practical use except for models with $|\mathcal{V}_\text{new}|$=50 or 100 models. Please see the discussions and recommendations in the paper.*
| Approach | my | si | te |
| --- | :---: | :---: | :---: |
| Random | [50](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-my-30K-50-rand-2x2ls-mtp-512) / [100](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-my-30K-rand-2x2ls-mtp-512) / [500](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-my-30K-500-rand-2x2ls-mtp-512) / [1000](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-my-30K-1000-rand-2x2ls-mtp-512) / [5000](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-my-30K-5000-rand-2x2ls-mtp-512) | [50](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-si-30K-50-rand-2x2ls-mtp-512) / [100](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-si-30K-rand-2x2ls-mtp-512) / [500](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-si-30K-500-rand-2x2ls-mtp-512) / [1000](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-si-30K-1000-rand-2x2ls-mtp-512) / [5000](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-si-30K-5000-rand-2x2ls-mtp-512) | [50](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-te-30K-50-rand-2x2ls-mtp-512) / [100](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-te-30K-rand-2x2ls-mtp-512) / [500](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-te-30K-500-rand-2x2ls-mtp-512) / [1000](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-te-30K-1000-rand-2x2ls-mtp-512) / [5000](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-te-30K-5000-rand-2x2ls-mtp-512) |
| Mean | [50](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-my-30K-50-mean-2x2ls-mtp-512) / [100](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-my-30K-mean-2x2ls-mtp-512) / [500](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-my-30K-500-mean-2x2ls-mtp-512) / [1000](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-my-30K-1000-mean-2x2ls-mtp-512) / [5000](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-my-30K-5000-mean-2x2ls-mtp-512) | [50](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-si-30K-50-mean-2x2ls-mtp-512) / [100](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-si-30K-mean-2x2ls-mtp-512) / [500](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-si-30K-500-mean-2x2ls-mtp-512) / [1000](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-si-30K-1000-mean-2x2ls-mtp-512) / [5000](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-si-30K-5000-mean-2x2ls-mtp-512) | [50](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-te-30K-50-mean-2x2ls-mtp-512) / [100](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-te-30K-mean-2x2ls-mtp-512) / [500](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-te-30K-500-mean-2x2ls-mtp-512) / [1000](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-te-30K-1000-mean-2x2ls-mtp-512) / [5000](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-te-30K-5000-mean-2x2ls-mtp-512) |
| Align | [50](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-my-30K-50-align-2x2ls-mtp-512) / [100](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-my-30K-align-2x2ls-mtp-512) / [500](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-my-30K-500-align-2x2ls-mtp-512) / [1000](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-my-30K-1000-align-2x2ls-mtp-512) / [5000](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-my-30K-5000-align-2x2ls-mtp-512) | [50](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-si-30K-50-align-2x2ls-mtp-512) / [100](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-si-30K-align-2x2ls-mtp-512) / [500](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-si-30K-500-align-2x2ls-mtp-512) / [1000](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-si-30K-1000-align-2x2ls-mtp-512) / [5000](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-si-30K-5000-align-2x2ls-mtp-512) | [50](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-te-30K-50-align-2x2ls-mtp-512) / [100](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-te-30K-align-2x2ls-mtp-512) / [500](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-te-30K-500-align-2x2ls-mtp-512) / [1000](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-te-30K-1000-align-2x2ls-mtp-512) / [5000](https://huggingface.co/atsuki-yamaguchi/Llama-2-7b-hf-te-30K-5000-align-2x2ls-mtp-512) |


### Llama3
#### Models adapted using 2x2 LS + MTP + 512
*You might be able to use these models for practical use.*
| Approach | Link |
| --- | --- |
| LAPT | [my](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-my-30K-lapt) / [si](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-si-30K-lapt) / [te](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-te-30K-lapt) |
| Random | [my](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-my-30K-100-rand) / [si](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-si-30K-100-rand) / [te](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-te-30K-100-rand) |
| Mean | [my](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-my-30K-100-mean) / [si](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-si-30K-100-mean) / [te](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-te-30K-100-mean) |
| Align | [my](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-my-30K-100-align) / [si](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-si-30K-100-align) / [te](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-te-30K-100-align) |


#### Models used for target vocabulary size analysis
*Not recommended for practical use except for models with $|\mathcal{V}_\text{new}|$=50 or 100 models. Please see the discussions and recommendations in the paper.*
| Approach | my | si | te |
| --- | :---: | :---: | :---: |
| Random | [50](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-my-30K-50-rand) / [100](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-my-30K-100-rand) / [500](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-my-30K-500-rand) / [1000](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-my-30K-1000-rand) / [5000](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-my-30K-5000-rand) | [50](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-si-30K-50-rand) / [100](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-si-30K-100-rand) / [500](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-si-30K-500-rand) / [1000](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-si-30K-1000-rand) / [5000](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-si-30K-5000-rand) | [50](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-te-30K-50-rand) / [100](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-te-30K-100-rand) / [500](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-te-30K-500-rand) / [1000](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-te-30K-1000-rand) / [5000](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-te-30K-5000-rand) |
| Mean | [50](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-my-30K-50-mean) / [100](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-my-30K-100-mean) / [500](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-my-30K-500-mean) / [1000](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-my-30K-1000-mean) / [5000](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-my-30K-5000-mean) | [50](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-si-30K-50-mean) / [100](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-si-30K-100-mean) / [500](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-si-30K-500-mean) / [1000](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-si-30K-1000-mean) / [5000](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-si-30K-5000-mean) | [50](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-te-30K-50-mean) / [100](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-te-30K-100-mean) / [500](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-te-30K-500-mean) / [1000](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-te-30K-1000-mean) / [5000](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-te-30K-5000-mean) |
| Align | [50](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-my-30K-50-align) / [100](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-my-30K-100-align) / [500](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-my-30K-500-align) / [1000](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-my-30K-1000-align) / [5000](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-my-30K-5000-align) | [50](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-si-30K-50-align) / [100](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-si-30K-100-align) / [500](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-si-30K-500-align) / [1000](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-si-30K-1000-align) / [5000](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-si-30K-5000-align) | [50](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-te-30K-50-align) / [100](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-te-30K-100-align) / [500](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-te-30K-500-align) / [1000](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-te-30K-1000-align) / [5000](https://huggingface.co/atsuki-yamaguchi/Llama-3-8B-te-30K-5000-align) |



### Gemma2
#### Models adapted using 2x2 LS + MTP + 512
*You might be able to use these models for practical use.*
| Approach | Link |
| --- | --- |
| LAPT | [my](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-my-30K-lapt) / [si](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-si-30K-lapt) / [te](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-te-30K-lapt) |
| Random | [my](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-my-30K-rand) / [si](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-si-30K-rand) / [te](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-te-30K-rand) |
| Mean | [my](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-my-30K-mean) / [si](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-si-30K-mean) / [te](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-te-30K-mean) |
| Align | [my](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-my-30K-align) / [si](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-si-30K-align) / [te](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-te-30K-align) |

#### Models used for target vocabulary size analysis
*Not recommended for practical use except for models with $|\mathcal{V}_\text{new}|$=50 or 100 models. Please see the discussions and recommendations in the paper.*
| Approach | my | si | te |
| --- | :---: | :---: | :---: |
| Random | [50](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-my-30K-50-rand) / [100](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-my-30K-rand) / [500](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-my-30K-500-rand) / [1000](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-my-30K-1000-rand) / [5000](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-my-30K-5000-rand) | [50](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-si-30K-50-rand) / [100](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-si-30K-rand) / [500](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-si-30K-500-rand) / [1000](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-si-30K-1000-rand) / [5000](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-si-30K-5000-rand) | [50](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-te-30K-50-rand) / [100](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-te-30K-rand) / [500](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-te-30K-500-rand) / [1000](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-te-30K-1000-rand) / [5000](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-te-30K-5000-rand) |
| Mean | [50](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-my-30K-50-mean) / [100](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-my-30K-mean) / [500](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-my-30K-500-mean) / [1000](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-my-30K-1000-mean) / [5000](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-my-30K-5000-mean) | [50](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-si-30K-50-mean) / [100](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-si-30K-mean) / [500](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-si-30K-500-mean) / [1000](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-si-30K-1000-mean) / [5000](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-si-30K-5000-mean) | [50](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-te-30K-50-mean) / [100](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-te-30K-mean) / [500](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-te-30K-500-mean) / [1000](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-te-30K-1000-mean) / [5000](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-te-30K-5000-mean) |
| Align | [50](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-my-30K-50-align) / [100](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-my-30K-align) / [500](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-my-30K-500-align) / [1000](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-my-30K-1000-align) / [5000](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-my-30K-5000-align) | [50](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-si-30K-50-align) / [100](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-si-30K-align) / [500](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-si-30K-500-align) / [1000](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-si-30K-1000-align) / [5000](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-si-30K-5000-align) | [50](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-te-30K-50-align) / [100](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-te-30K-align) / [500](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-te-30K-500-align) / [1000](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-te-30K-1000-align) / [5000](https://huggingface.co/atsuki-yamaguchi/gemma-2-9b-te-30K-5000-align) |



## License
This code is licensed under the MIT License. The models are licensed under the respective licenses of the original models. Please refer to the Hugging Face Model Hub for the licenses of the models.

## Citation
If you use this code or models, please cite the following paper.
```
@article{yamaguchi-etal-2024-effectively,
    title={How Can We Effectively Expand the Vocabulary of LLMs with 0.01GB of Target Language Text?}, 
    author={Atsuki Yamaguchi and Aline Villavicencio and Nikolaos Aletras},
    year={2024},
    journal={ArXiv},
    year={2024},
    volume={abs/2406.11477},
    url={https://arxiv.org/abs/2406.11477}, 
}
```
