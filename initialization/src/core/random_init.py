
import json
import math
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def round_to_nearest_multiple(vocabulary_size, multiple):
    rounded_size = math.ceil(vocabulary_size / multiple) * multiple
    return rounded_size

def instantiate_model_by_random(
    source_model: AutoModelForCausalLM,
    source_tokenizer: AutoTokenizer,
    target_tokenizer: AutoTokenizer,
    tie_word_embeddings: bool = False
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    # expand the embeddings
    source_embeddings = source_model.get_input_embeddings().weight.detach().numpy()
    target_embeddings = np.random.normal(
        np.mean(source_embeddings, axis=0), 
        np.std(source_embeddings, axis=0), 
        (round_to_nearest_multiple(len(target_tokenizer), 8), 
         source_embeddings.shape[1])
    )
    target_embeddings[:source_embeddings.shape[0]] = source_embeddings
    
    if not tie_word_embeddings:
        print("You are using the output projection init.")
        source_head_embeddings = source_model.get_output_embeddings().weight.detach().numpy()
        target_head_embeddings = np.random.normal(
            np.mean(source_head_embeddings, axis=0), 
            np.std(source_head_embeddings, axis=0), 
            (round_to_nearest_multiple(len(target_tokenizer), 8), 
             source_head_embeddings.shape[1])
        )
        target_head_embeddings[:source_head_embeddings.shape[0]] = source_head_embeddings
    
    # set weights
    target_model = source_model
    target_model.resize_token_embeddings(
        len(target_tokenizer), 
        pad_to_multiple_of=8 # See https://github.com/huggingface/transformers/issues/26303
    )
    target_model.get_input_embeddings().weight.data = torch.from_numpy(target_embeddings)
    target_model.config.vocab_size = round_to_nearest_multiple(len(target_tokenizer), 8)
    if not tie_word_embeddings:
        target_model.get_output_embeddings().weight.data = torch.from_numpy(target_head_embeddings)
    else:
        target_model.tie_weights()
    
    return target_model, target_tokenizer
