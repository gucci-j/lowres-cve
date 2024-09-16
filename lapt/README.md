Language adaptive pre-training
===

Here, we conduct continual pre-training. For reproduction, you just need to run the provided scripts. Note that you need to modify the paths starting with `/path/to/` in each script. 

The following scripts are to train models with our best training strategies: 2x2LS + MTP + 512. If you would like to train models with other training strategies, you can modify the training scripts accordingly.

* Llama2: [llama2.sh](./scripts/llama2.sh)
* Llama3: [llama3.sh](./scripts/llama3.sh)
* Gemma2: [gemma2.sh](./scripts/gemma2.sh)
