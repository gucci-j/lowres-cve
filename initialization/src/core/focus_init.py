import math

import entmax
import fasttext
import numpy as np
import torch
from fastdist import fastdist
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def round_to_nearest_multiple(vocabulary_size, multiple) -> int:
    rounded_size = math.ceil(vocabulary_size / multiple) * multiple
    return rounded_size


def is_very_rare_token(
    token: str,
    fasttext_model: dict[str, np.ndarray]
) -> bool:
    return token not in fasttext_model or np.absolute(fasttext_model[token]).sum() == 0


def instantiate_model_by_focus(
    source_model: AutoModelForCausalLM,
    source_tokenizer: AutoTokenizer,
    target_tokenizer: AutoTokenizer,
    fasttext_model_path: str,
    tie_word_embeddings: bool = False,
    is_word_level: bool = False,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    # init
    fasttext_model = fasttext.load_model(fasttext_model_path)
    if is_word_level:
        fasttext_token_embs = {}
        for token, idx in target_tokenizer.get_vocab().items():
            clean_token = target_tokenizer.decode(idx).strip()
            fasttext_token_embs[token] = fasttext_model.get_word_vector(clean_token)
        fasttext_model = fasttext_token_embs
    source_embeddings = source_model.get_input_embeddings().weight.detach().numpy()
    target_embeddings = np.random.normal(
        np.mean(source_embeddings, axis=0), 
        np.std(source_embeddings, axis=0), 
        (round_to_nearest_multiple(len(target_tokenizer), 8), 
         source_embeddings.shape[1])
    )
    target_embeddings[:len(source_tokenizer)] = source_embeddings
    if not tie_word_embeddings:
        source_head_embeddings = source_model.get_output_embeddings().weight.detach().numpy()
        target_head_embeddings = np.random.normal(
            np.mean(source_head_embeddings, axis=0), 
            np.std(source_head_embeddings, axis=0), 
            (round_to_nearest_multiple(len(target_tokenizer), 8), 
             source_head_embeddings.shape[1])
        )
        target_head_embeddings[:len(source_tokenizer)] = source_head_embeddings
    else:
        source_head_embeddings = None
        target_head_embeddings = None
    overlapping_tokens = set(source_tokenizer.get_vocab().keys()) & set(target_tokenizer.get_vocab().keys())
    new_tokens = set(target_tokenizer.get_vocab().keys()) - overlapping_tokens
    target_token_to_idx = {t: i for t, i in target_tokenizer.get_vocab().items()}
    source_token_to_idx = {t: i for t, i in source_tokenizer.get_vocab().items()}
    
    #####
    # Generate the auxiliary embeddings for overlapping tokens
    #####
    token_to_auxiliary_embeddings = {}
    temp_overlapping_tokens_list = []
    for token in list(overlapping_tokens):
        if is_very_rare_token(token, fasttext_model):
            continue
        else:
            token_to_auxiliary_embeddings[token] = fasttext_model[token]
            temp_overlapping_tokens_list.append(token)
    overlapping_tokens_list = temp_overlapping_tokens_list
    
    #####
    # Initialize the missing tokens with FOCUS
    #####
    # Generate the auxiliary embeddings for missing tokens
    temp_missing_tokens_list = []
    for token in list(new_tokens):
        if is_very_rare_token(token, fasttext_model):
            continue
        else:
            token_to_auxiliary_embeddings[token] = fasttext_model[token]
            temp_missing_tokens_list.append(token)
    missing_tokens_list = temp_missing_tokens_list
    
    #####
    # Get the embeddings for the missing tokens and overlapping tokens in FastText
    #####
    missing_auxiliary_embedding_matrix = np.asarray(
        [token_to_auxiliary_embeddings[t] for t in missing_tokens_list],
        dtype="float32"
    ) # -> (len(missing_tokens), fasttext_embedding_dim)
    overlapping_auxiliary_embedding_matrix = np.asarray(
        [token_to_auxiliary_embeddings[t] for t in overlapping_tokens_list],
        dtype="float32"
    ) # -> (len(overlapping_tokens), fasttext_embedding_dim)

    #####
    # Compute the cosine similarity between the missing tokens and overlapping tokens in FastText
    #####
    cos_sims = fastdist.cosine_matrix_to_matrix(
        missing_auxiliary_embedding_matrix,
        overlapping_auxiliary_embedding_matrix,
    ) # -> (len(missing_tokens), len(overlapping_tokens))
    # Not needed anymore, save memory
    del missing_auxiliary_embedding_matrix
    del overlapping_auxiliary_embedding_matrix

    #####
    # Compute the weighted mean of the overlapping tokens in the source model
    #####
    # Get the embeddings for the overlapping tokens in the source model
    overlapping_tokens_idxs = \
        [source_token_to_idx[t] for t in overlapping_tokens_list]
    overlapping_token_vecs = torch.from_numpy(
        source_embeddings[overlapping_tokens_idxs, :]
    ) # -> (len(overlapping_tokens), source_embedding_dim)
    if not tie_word_embeddings:
        overlapping_head_vecs = torch.from_numpy(
            source_head_embeddings[overlapping_tokens_idxs, :]
        )

    # Initialize the target embeddings with the weighted mean of the overlapping tokens in the source model
    for index, token in enumerate(tqdm(missing_tokens_list)):
        # Get the cosine similarity scores for the missing token
        token_cos_sim = entmax.sparsemax(
            torch.from_numpy(cos_sims[index])
        ) # -> (len(overlapping_tokens),)
        
        # Get the weighted mean of the overlapping tokens in the source model
        mask = token_cos_sim > 0.0
        masked_token_cos_sim = token_cos_sim[mask] # -> (num_token_cos_sim_positive,)
        masked_overlapping_token_vecs = overlapping_token_vecs[mask] # -> (num_token_cos_sim_positive, source_embedding_dim)
        weighted_src_embs = torch.mul(
            masked_overlapping_token_vecs, 
            masked_token_cos_sim.unsqueeze(1)
        ) # -> (num_token_cos_sim_positive, source_embedding_dim)
        weighted_mean = torch.sum(weighted_src_embs, dim=0) # -> (source_embedding_dim,)
        if not tie_word_embeddings:
            masked_overlapping_head_vecs = overlapping_head_vecs[mask]
            weighted_head_embs = torch.mul(
                masked_overlapping_head_vecs,
                masked_token_cos_sim.unsqueeze(1)
            )
            weighted_head_mean = torch.sum(weighted_head_embs, dim=0)
        
        # Set the embedding of the current missing token to the weighted mean
        target_embeddings[target_token_to_idx[token]] = weighted_mean.detach().numpy()
        if not tie_word_embeddings:
            target_head_embeddings[target_token_to_idx[token]] = weighted_head_mean.detach().numpy()
    
    # finalize
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
