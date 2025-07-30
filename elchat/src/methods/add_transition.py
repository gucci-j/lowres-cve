from typing import Union

import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from .core import linear, slerp

def add_transition(
    model_adapted: AutoModelForCausalLM, 
    model_instruct: AutoModelForCausalLM, 
    transition_indices: list[int] = [2, 3, -4, -3],
    transition_rates: list[float] = [0.5, 0.5, 0.5, 0.5],
    transition_method: str = "linear",
    **kwargs
) -> AutoModelForCausalLM:
    # Sanity check
    assert model_adapted.config.model_type == model_instruct.config.model_type, "The model types of the two models must be the same!"

    # Mix weights for transition layers
    print(f"\t[add_transition]: Mixing weights for transition layers for {transition_indices} layers...")
    print(f"\t[add_transition]: Transition method: {transition_method}")
    if model_adapted.config.model_type in ("llama", "qwen2", "qwen3"):
        layer_attr_names = [
            "self_attn",
            "mlp",
            "input_layernorm",
            "post_attention_layernorm",
        ]
    elif model_adapted.config.model_type == "gemma2":
        layer_attr_names = [
            "self_attn",
            "mlp",
            "input_layernorm",
            "post_attention_layernorm",
            "post_feedforward_layernorm",
            "pre_feedforward_layernorm",    
        ]
    else:
        raise NotImplementedError
    for index in transition_indices:
        print(f"\t[add_transition]: Now mixing the {index} layer with a transition rate of {transition_rates[index]}...")
        with torch.no_grad():
            for attr_name in layer_attr_names:
                if attr_name == "self_attn":
                    # Mix the weights for the self-attention layer
                    for attr_name2 in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                        if transition_method == "linear":
                            getattr(getattr(model_instruct.model.layers[index], attr_name), attr_name2).weight.copy_(
                                linear(
                                    transition_rates[index],
                                    getattr(getattr(model_adapted.model.layers[index], attr_name), attr_name2).weight,
                                    getattr(getattr(model_instruct.model.layers[index], attr_name), attr_name2).weight
                                )
                            )
                        elif transition_method == "slerp":
                            getattr(getattr(model_instruct.model.layers[index], attr_name), attr_name2).weight.copy_(
                                slerp(
                                    transition_rates[index],
                                    getattr(getattr(model_adapted.model.layers[index], attr_name), attr_name2).weight,
                                    getattr(getattr(model_instruct.model.layers[index], attr_name), attr_name2).weight
                                )
                            )
                        else:
                            raise NotImplementedError
                elif attr_name == "mlp":
                    # Mix the weights for the MLP layer
                    for attr_name2 in ["gate_proj", "up_proj", "down_proj"]:
                        if transition_method == "linear":
                            getattr(getattr(model_instruct.model.layers[index], attr_name), attr_name2).weight.copy_(
                                linear(
                                    transition_rates[index],
                                    getattr(getattr(model_adapted.model.layers[index], attr_name), attr_name2).weight,
                                    getattr(getattr(model_instruct.model.layers[index], attr_name), attr_name2).weight
                                )
                            )
                        elif transition_method == "slerp":
                            getattr(getattr(model_instruct.model.layers[index], attr_name), attr_name2).weight.copy_(
                                slerp(
                                    transition_rates[index],
                                    getattr(getattr(model_adapted.model.layers[index], attr_name), attr_name2).weight,
                                    getattr(getattr(model_instruct.model.layers[index], attr_name), attr_name2).weight
                                )
                            )
                        else:
                            raise NotImplementedError
                else:
                    # Mix the weights for the other layers
                    if transition_method == "linear":
                        getattr(model_instruct.model.layers[index], attr_name).weight.copy_(
                            linear(
                                transition_rates[index],
                                getattr(model_adapted.model.layers[index], attr_name).weight,
                                getattr(model_instruct.model.layers[index], attr_name).weight
                            )
                        )
                    elif transition_method == "slerp":
                        getattr(model_instruct.model.layers[index], attr_name).weight.copy_(
                            slerp(
                                transition_rates[index],
                                getattr(model_adapted.model.layers[index], attr_name).weight,
                                getattr(model_instruct.model.layers[index], attr_name).weight
                            )
                        )
                    else:
                        raise NotImplementedError
    
    print("\t[add_transition]: Done!")
    return model_instruct