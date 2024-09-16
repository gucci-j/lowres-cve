
import json
import math
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def round_to_nearest_multiple(vocabulary_size, multiple):
    rounded_size = math.ceil(vocabulary_size / multiple) * multiple
    return rounded_size

def instantiate_model_by_merge(
    source_model: AutoModelForCausalLM,
    source_tokenizer: AutoTokenizer,
    target_tokenizer: AutoTokenizer,
    tie_word_embeddings: bool = False
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    # init
    target_model_state = json.loads(target_tokenizer.backend_tokenizer.model.__getstate__())
    source_vocab = source_tokenizer.get_vocab()
    target_vocab = target_tokenizer.get_vocab()
    
    # get new vocab
    new_vocab = []
    for token in target_model_state["vocab"]:
        if token not in source_vocab:
            new_vocab.append(token)

    # identify new merges
    new_token_to_old_tokens = {}
    new_token_to_merge_indices = {}
    tmp_merges = []
    for index, merge in enumerate(target_model_state["merges"]):
        token_1, token_2 = merge.split(" ")
        token = token_1 + token_2
        if token in new_vocab:
            if token_1 not in new_vocab and token_2 not in new_vocab:
                # can map to existing tokens
                if new_token_to_merge_indices.get(token) is not None:
                    new_token_to_merge_indices[token].append(index)
                else:
                    new_token_to_merge_indices[token] = [index]
            else:
                # completely new
                tmp_merges.append((index, merge))

    # map new token to old tokens
    for token, indices in new_token_to_merge_indices.items():
        for index in indices:
            token_1, token_2 = target_model_state["merges"][index].split(" ")
            if new_token_to_old_tokens.get(token) is None:
                new_token_to_old_tokens[token] = [[source_vocab[token_1], source_vocab[token_2]]]
            else:
                new_token_to_old_tokens[token].append([source_vocab[token_1], source_vocab[token_2]])
    
    # process unprocessed merges
    def convert_merge_into_old_tokens(merge) -> bool:
        # Given a merge, return token indices
        token_1, token_2 = merge.split(" ")
        token = token_1 + token_2
        if token_1 in new_vocab and token_2 not in new_vocab: # -> new_merge + old
            if new_token_to_old_tokens.get(token_1) is not None: # -> have a map to existing tokens
                if new_token_to_old_tokens.get(token) is None:
                    new_token_to_old_tokens[token] = [[new_token_to_old_tokens[token_1], source_vocab[token_2]]]
                else:
                    new_token_to_old_tokens[token].append([new_token_to_old_tokens[token_1], source_vocab[token_2]])
                return True
            else: # -> have no map to existing tokens
                return False
        elif token_1 not in new_vocab and token_2 in new_vocab: # -> old + new_merge 
            if new_token_to_old_tokens.get(token_2) is not None: # -> have a map to existing tokens
                if new_token_to_old_tokens.get(token) is None:
                    new_token_to_old_tokens[token] = [[source_vocab[token_1], new_token_to_old_tokens[token_2]]]
                else:
                    new_token_to_old_tokens[token].append([source_vocab[token_1], new_token_to_old_tokens[token_2]])
                return True
            else: # -> have no map to existing tokens
                return False
        
        elif token_1 in new_vocab and token_2 in new_vocab: # -> new + new
            if new_token_to_old_tokens.get(token_1) is not None and new_token_to_old_tokens.get(token_2) is not None:
                if new_token_to_old_tokens.get(token) is None:
                    new_token_to_old_tokens[token] = [[new_token_to_old_tokens[token_1], new_token_to_old_tokens[token_2]]]
                else:
                    new_token_to_old_tokens[token].append([new_token_to_old_tokens[token_1], new_token_to_old_tokens[token_2]])
                return True
            else:
                return False
    
    while True:
        unmerged = []
        for index, merge in tmp_merges:
            if not convert_merge_into_old_tokens(merge):
                unmerged.append((index, merge))
        if len(unmerged) == 0:
            break
        if len(unmerged) == len(tmp_merges):
            # if there is only one merge and it is unmerged, then it is a completely new merge
            unmerged_index, unmerged_merge = unmerged[0]
            token_1, token_2 = unmerged_merge.split(" ")
            token = token_1 + token_2
            if token_1 in new_vocab and token_2 not in new_vocab: # -> new_merge + old
                new_token_to_old_tokens[token_1] = [source_tokenizer(token_1, add_special_tokens=False).input_ids]
            elif token_1 not in new_vocab and token_2 in new_vocab: # -> old + new_merge
                pass
            elif token_1 in new_vocab and token_2 in new_vocab: # -> new + new
                pass
            break
        else:
            tmp_merges = unmerged
    
    # expand the embeddings
    source_embeddings = source_model.get_input_embeddings().weight.detach().numpy()
    target_embeddings = np.random.normal(
        np.mean(source_embeddings, axis=0), 
        np.std(source_embeddings, axis=0), 
        (round_to_nearest_multiple(len(target_tokenizer), 8), 
         source_embeddings.shape[1])
    )
    target_embeddings[:len(source_tokenizer)] = source_embeddings
    
    def compute_embeddings(indices) -> np.ndarray:
        if type(indices) == int:
            return source_embeddings[indices]
        else:
            embeds = []
            for index in indices:
                embeds.append(compute_embeddings(index))
            return np.mean(np.array(embeds), axis=0)
    
    for token, indices in new_token_to_old_tokens.items():
        embeds = []
        for index in indices:
            embeds.append(compute_embeddings(index))
        target_embeddings[
            target_vocab[token]
        ] = np.mean(np.array(embeds), axis=0)
    
    # init output projection
    if not tie_word_embeddings:
        print("You are using the output projection init.")
        source_head_embeddings = source_model.get_output_embeddings().weight.detach().numpy()
        target_head_embeddings = np.zeros(
            (round_to_nearest_multiple(len(target_tokenizer), 8), 
            source_head_embeddings.shape[1])
        )
        target_head_embeddings[:len(source_tokenizer)] = source_head_embeddings
        
        def compute_head_embeddings(indices) -> np.ndarray:
            if type(indices) == int:
                return source_head_embeddings[indices]
            else:
                embeds = []
                for index in indices:
                    embeds.append(compute_embeddings(index))
                return np.mean(np.array(embeds), axis=0)

        for token, indices in new_token_to_old_tokens.items():
            embeds = []
            for index in indices:
                embeds.append(compute_head_embeddings(index))
            target_head_embeddings[
                target_vocab[token]
            ] = np.mean(np.array(embeds), axis=0)
    
    # set weights
    target_model = source_model
    target_model.resize_token_embeddings(
        len(target_tokenizer), 
        pad_to_multiple_of=8 # See https://github.com/huggingface/transformers/issues/26303
    )
    target_model.get_input_embeddings().weight.data = torch.from_numpy(target_embeddings)
    print(target_embeddings.shape)
    print(target_model)
    target_model.config.vocab_size = round_to_nearest_multiple(len(target_tokenizer), 8)
    if not tie_word_embeddings:
        target_model.get_output_embeddings().weight.data = torch.from_numpy(target_head_embeddings)
    else:
        target_model.tie_weights()
        
    return target_model, target_tokenizer
