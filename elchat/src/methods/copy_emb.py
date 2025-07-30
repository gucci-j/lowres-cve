import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

def copy_emb(
    model_adapted: AutoModelForCausalLM, 
    model_adapted_base: AutoModelForCausalLM,
    model_instruct: AutoModelForCausalLM, 
    tokenizer_adapted: AutoTokenizer,
    tokenizer_instruct: AutoTokenizer,
    consider_special_tokens: bool = False,
    consider_added_tokens: bool = False,
    copy_new_tokens_only: bool = False,
    tie_weights: bool = False,
    **kwargs
) -> AutoModelForCausalLM:
    # Sanity check
    assert model_adapted.config.model_type == model_instruct.config.model_type, "The model types of the two models must be the same!"
    
    # Resize and copy the embedding layer
    print("\t[copy_emb]: Resizing and copying the embedding layer...")
    model_instruct.resize_token_embeddings(
        len(tokenizer_adapted),
        pad_to_multiple_of=8
    )
    if model_adapted_base is not None:
        model_instruct.config.vocab_size = model_adapted_base.config.vocab_size
        with torch.no_grad():
            embeddings_adapted = model_adapted_base.get_input_embeddings().weight.detach().numpy()
            embeddings_instruct = model_instruct.get_input_embeddings().weight.detach().numpy()
            if copy_new_tokens_only:
                print("\t[copy_emb]: Only copy new token weights...")
                embeddings_instruct[len(tokenizer_instruct):] = embeddings_adapted[len(tokenizer_instruct):]
                model_instruct.get_input_embeddings().weight.data = torch.from_numpy(embeddings_instruct)
            else:
                print("\t[copy_emb]: Copy all token weights...")
                if consider_special_tokens:
                    print("\t[copy_emb]: Put back special token weights...")
                    all_special_tokens = tokenizer_instruct.all_special_tokens
                    vocab = tokenizer_instruct.get_vocab()
                    all_special_tokens_indices = [vocab[token] for token in all_special_tokens]
                    embeddings_adapted[all_special_tokens_indices] = embeddings_instruct[all_special_tokens_indices]
                if consider_added_tokens:
                    print("\t[copy_emb]: Put back special added token weights...")
                    added_vocab = tokenizer_instruct.get_added_vocab()
                    added_tokens_indices = [index for index in added_vocab.values()]
                    embeddings_adapted[added_tokens_indices] = embeddings_instruct[added_tokens_indices]
                model_instruct.get_input_embeddings().weight.data = torch.from_numpy(embeddings_adapted)
            del embeddings_adapted
            del embeddings_instruct

            if tie_weights:
                model_instruct.tie_weights()
            else:
                lm_head_adapted = model_adapted_base.get_output_embeddings().weight.detach().numpy()
                lm_head_instruct = model_instruct.get_output_embeddings().weight.detach().numpy()
                if copy_new_tokens_only:
                    lm_head_instruct[len(tokenizer_instruct):] = lm_head_adapted[len(tokenizer_instruct):]
                    model_instruct.get_output_embeddings().weight.data = torch.from_numpy(lm_head_instruct)
                else:
                    if consider_special_tokens:
                        lm_head_adapted[all_special_tokens_indices] = lm_head_instruct[all_special_tokens_indices]
                    if consider_added_tokens:
                        lm_head_adapted[added_tokens_indices] = lm_head_instruct[added_tokens_indices]
                    model_instruct.get_output_embeddings().weight.data = torch.from_numpy(lm_head_adapted)
                del lm_head_adapted
                del lm_head_instruct
                
    else:
        model_instruct.config.vocab_size = model_adapted.config.vocab_size
        with torch.no_grad():
            embeddings_adapted = model_adapted.get_input_embeddings().weight.detach().numpy()
            embeddings_instruct = model_instruct.get_input_embeddings().weight.detach().numpy()
            if copy_new_tokens_only:
                print("\t[copy_emb]: Only copy new token weights...")
                embeddings_instruct[len(tokenizer_instruct):] = embeddings_adapted[len(tokenizer_instruct):]
                model_instruct.get_input_embeddings().weight.data = torch.from_numpy(embeddings_instruct)
            else:
                print("\t[copy_emb]: Copy all token weights...")
                if consider_special_tokens:
                    print("\t[copy_emb]: Put back special token weights...")
                    all_special_tokens = tokenizer_instruct.all_special_tokens
                    vocab = tokenizer_instruct.get_vocab()
                    all_special_tokens_indices = [vocab[token] for token in all_special_tokens]
                    embeddings_adapted[all_special_tokens_indices] = embeddings_instruct[all_special_tokens_indices]
                if consider_added_tokens:
                    print("\t[copy_emb]: Put back special added token weights...")
                    added_vocab = tokenizer_instruct.get_added_vocab()
                    added_tokens_indices = [index for index in added_vocab.values()]
                    embeddings_adapted[added_tokens_indices] = embeddings_instruct[added_tokens_indices]
                model_instruct.get_input_embeddings().weight.data = torch.from_numpy(embeddings_adapted)
            del embeddings_adapted
            del embeddings_instruct

            if tie_weights:
                model_instruct.tie_weights()
            else:
                lm_head_adapted = model_adapted.get_output_embeddings().weight.detach().numpy()
                lm_head_instruct = model_instruct.get_output_embeddings().weight.detach().numpy()
                if copy_new_tokens_only:
                    lm_head_instruct[len(tokenizer_instruct):] = lm_head_adapted[len(tokenizer_instruct):]
                    model_instruct.get_output_embeddings().weight.data = torch.from_numpy(lm_head_instruct)
                else:
                    if consider_special_tokens:
                        lm_head_adapted[all_special_tokens_indices] = lm_head_instruct[all_special_tokens_indices]
                    if consider_added_tokens:
                        lm_head_adapted[added_tokens_indices] = lm_head_instruct[added_tokens_indices]
                    model_instruct.get_output_embeddings().weight.data = torch.from_numpy(lm_head_adapted)
                del lm_head_adapted
                del lm_head_instruct
        
    print("\t[copy_emb]: Done!")
    return model_instruct