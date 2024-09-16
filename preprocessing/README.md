Preprocessing
===

Here, we generate LAPT and evaluation data for Llama2, Llama3, and Gemma2. Also, we train tokenizers and fastText models.

## Reproduction
### 1. Download CC-100 data
Download the CC-100 data for 10 languages: Arabic, Burmese, German, Greek, Hindi, Japanese, Sinhala, Swahili, Telugu, and Thai. They are available at https://data.statmt.org/cc-100/

### 2. Run provided scripts
Run the following scripts. Note that you need to modify the paths starting with `/path/to/` in each script.

* Llama2: [llama2.sh](./scripts/llama2.sh)
* Llama3: [llama3.sh](./scripts/llama3.sh)
* Gemma2: [gemma2.sh](./scripts/gemma2.sh)
