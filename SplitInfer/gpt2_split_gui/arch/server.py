from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from transformers.utils import logging
from transformers.models.gpt2.modeling_gpt2 import GPT2PreTrainedModel, GPT2Model, GPT2LMHeadModel

logger = logging.get_logger(__name__)

class GPT2ModelServer(GPT2PreTrainedModel):
    def __init__(self, model: GPT2Model, cut_block=2):
        super().__init__(model.config)
        
        self.config = model.config
        self.cut_block = cut_block
        self.embed_dim = self.config.hidden_size
        self.wte = model.wte
        self.wpe = model.wpe

        self.h = model.h[cut_block:]
        self.ln_f = model.ln_f

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False
        self._attn_implementation = self.config._attn_implementation

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def forward(
        self,
        hidden_states, 
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_shape = (-1,) + (hidden_states.shape[1],) + (hidden_states.size(-1),)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # print(f"After setting output_attentions: {output_attentions}")

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # print(f"After setting output_hidden_states: {output_hidden_states}")

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # print(f"After setting use_cache: {use_cache}")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print(f"After setting return_dict: {return_dict}")

        batch_size = hidden_states.shape[0]

        device = hidden_states.device if hidden_states is not None else hidden_states.device
        # print(f"Device determined: {device}")

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
            # print(f"No past key values, set past length to {past_length}")
        else:
            past_length = past_key_values[0][0].size(-2)
            # print(f"Past length determined from past key values: {past_length}")

        # Attention mask.
        _use_sdpa = self._attn_implementation == "sdpa" and output_attentions is False and head_mask is None
        attention_mask = attention_mask.view(batch_size, -1) if attention_mask is not None else None
        # print(f"Attention mask reshaped: {attention_mask if attention_mask is not None else 'None'}")

        if self._attn_implementation == "flash_attention_2":
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            # print(f"Attention mask filtered for flash attention: {attention_mask}")

        elif _use_sdpa:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask=attention_mask,
                input_shape=(batch_size, hidden_states.shape[-2]),
                inputs_embeds=hidden_states,
                past_key_values_length=past_length,
            )
            # print(f"Attention mask prepared for SDPA: {attention_mask if attention_mask is not None else 'None'}")

        else:
            if attention_mask is not None:
                # We create a 3D attention mask from a 2D tensor mask.
                # Sizes are [batch_size, 1, 1, to_seq_length]
                # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
                # this attention mask is more simple than the triangular masking of causal attention
                # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
                attention_mask = attention_mask[:, None, None, :]

                # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
                # masked positions, this operation will create a tensor which is 0.0 for
                # positions we want to attend and the dtype's smallest value for masked positions.
                # Since we are adding it to the raw scores before the softmax, this is
                # effectively the same as removing these entirely.
                attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
                attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
                # print(f"Attention mask final shape: {attention_mask}, min value: {torch.min(attention_mask)}")

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            # print(f"Encoder hidden shape: {encoder_hidden_shape}")

            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
                # print(f"Created encoder attention mask: {encoder_attention_mask}")

            if _use_sdpa:
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    mask=encoder_attention_mask, dtype=hidden_states.dtype, tgt_len=hidden_states.shape[-2]
                )
                # print(f"Prepared encoder attention mask for SDPA: {encoder_attention_mask}")

            elif not self._attn_implementation == "flash_attention_2":
                encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
                # print(f"Inverted encoder attention mask: {encoder_attention_mask}")

        else:
            encoder_attention_mask = None
            # print(f"No cross attention, encoder attention mask set to None.")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer - self.cut_block)
        # print(f"Head mask obtained: {head_mask if head_mask is not None else 'None'}")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
            # print(f"Gradient checkpointing enabled, use_cache set to {use_cache}.")

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Model parallel
            if self.model_parallel:
                # print(f"Set device to {hidden_states.device} for model parallelism.")
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                # print(f"Collected hidden states: {len(all_hidden_states)}")

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            # print(f"Updated hidden states from block {i}: {hidden_states.sum()}, {hidden_states.shape}")

            if use_cache is True:
                presents = presents + (outputs[1],)
                # print(f"Collected present key values: {len(presents)}")

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                # print(f"Collected self attentions: {len(all_self_attentions)}")

                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)
                    # print(f"Collected cross attentions: {len(all_cross_attentions)}")

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))
                        # print(f"Moved hidden states to cuda:{k+1}.")

        hidden_states = self.ln_f(hidden_states)
        # print(f"Normalized hidden states: {hidden_states.sum()}, {hidden_states.shape}")

        hidden_states = hidden_states.view(output_shape)
        # print(f"Reshaped hidden states: {hidden_states.sum()}, {hidden_states.shape}")

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            # print(f"Collected final hidden states: {len(all_hidden_states)}")

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
    

class GPT2LMHeadModelServer(GPT2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, model: GPT2LMHeadModel, cut_block=2):
        super().__init__(model.config)

        self.config = model.config
        self.transformer = GPT2ModelServer(model.transformer, cut_block)
        self.lm_head = model.lm_head

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.cache_feature = dict()

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        hidden_states, 
        task_id,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print(f"After setting return_dict: {return_dict}")

        if task_id in self.cache_feature:
            hidden_states = torch.cat((self.cache_feature[task_id], hidden_states), dim=1)
        self.cache_feature[task_id] = hidden_states

        transformer_outputs = self.transformer(
            hidden_states, 
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        # print("Hidden states from transformer:", hidden_states.sum(), hidden_states.shape)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            # print(f"Set device to {self.transformer.first_device} for model parallelism")
            hidden_states = hidden_states.to(self.lm_head.weight.device)
            # print(f"Moved hidden states to {self.lm_head.weight.device}")

        lm_logits = self.lm_head(hidden_states)
        # print("Logits from language model head:", lm_logits.sum(), hidden_states.shape)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )

    def task_complete(self, task_id):
        if task_id in self.cache_feature:
            del self.cache_feature[task_id]