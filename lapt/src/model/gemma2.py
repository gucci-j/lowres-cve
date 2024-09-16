from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import Gemma2ForCausalLM
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Gemma2ForMultiCausalLM(Gemma2ForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    
    def __init__(self, config):
        super().__init__(config)
        self.num_lm_heads = config.num_lm_heads
        self.lm_heads = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.vocab_size, bias=None) for _ in range(config.num_lm_heads)]
        )
        #####
        # This is no longer needed
        #####
        if config.copy_lm_head:
            logger.info(f"Copying LM head weights...")
            for i in range(config.num_lm_heads):
                with torch.no_grad():
                    self.lm_heads[i].weight.copy_(self.lm_head.weight)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, GemmaForCausalLM

        >>> model = GemmaForCausalLM.from_pretrained("google/gemma-2-9b")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

        >>> prompt = "What is your favorite condiment?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "What is your favorite condiment?"
        ```"""
        if self.training and self.config._attn_implementation != "eager":
            logger.warning_once(
                "It is strongly recommended to train Gemma2 models with the `eager` attention implementation "
                f"instead of `{self.config._attn_implementation}`. Use `eager` with `AutoModelForCausalLM.from_pretrained('<path-to-checkpoint>', attn_implementation='eager')`."
            )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        # Source LM head
        hidden_states = outputs[0]
        
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping
        logits = logits.float()

        loss = None
        ntp_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            for i in range(0, self.num_lm_heads + 1):
                if i == 0: # source lm head
                    # Shift so that tokens < n predict n
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    # Flatten the tokens
                    shift_logits = shift_logits.view(-1, self.vocab_size)
                    shift_labels = shift_labels.view(-1)
                    # Enable model parallelism
                    shift_labels = shift_labels.to(shift_logits.device)
                    loss = loss_fct(shift_logits, shift_labels)
                    ntp_loss = loss.detach()
                    
                else:
                    nitp_logits = self.lm_heads[i - 1](hidden_states)
                    nitp_logits = nitp_logits.float()
                    # Shift so that tokens < n predict n
                    shift_logits = nitp_logits[..., :-(i+1), :].contiguous() # Removes the last i + 1 elements from the logits
                    shift_labels = labels[..., (i+1):].contiguous() # Removes the first i + 1 elements from the labels
                    # Flatten the tokens
                    shift_logits = shift_logits.view(-1, self.vocab_size)
                    shift_labels = shift_labels.view(-1)
                    # Enable model parallelism
                    shift_labels = shift_labels.to(shift_logits.device)
                    mtp_loss = loss_fct(shift_logits, shift_labels)
                    # Enable model parallelism again
                    loss = loss + mtp_loss.to(loss.device)

                    assert torch.isnan(shift_logits).sum().item() == 0, "NaN detected in shift_logits in MTP"
                    assert torch.isinf(shift_logits).sum().item() == 0, "Inf detected in shift_logits in MTP"

        if not return_dict:
            loss = None
            output = (None,) + outputs[1:]
            return (None,) + output if loss is not None else output
        logger.info(f"NTP loss: {ntp_loss}")

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )