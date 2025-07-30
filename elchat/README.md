ElChat: A post-hoc, training-free method for mitigating catastrophic forgetting in vocabulary expansion in LLMs
===

Here, we apply a post-hoc, training-free method to mitigate catastrophic forgetting in vocabulary expansion in LLMs. We use the ElChat method proposed by Yamaguchi et al. (2025). See https://github.com/gucci-j/chat-cve for more details.

## Reproduction
Run [`gemma2.sh`](scripts/gemma2.sh) to reproduce the results in the paper. The script will download the Gemma2 adapted model and run the ElChat method on it.
