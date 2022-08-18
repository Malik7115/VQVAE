import os
os.system('clear')

import einops

import torch 
import torch.nn as nn

from transformers import GPT2Config, GPT2PreTrainedModel, LogitsProcessorList
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.utils.model_parallel_utils import get_device_map, assert_device_map


from hf_ar.model import GPTModelCIFAR
from dataset import LMDBDataset



class GPT2InferenceModel(GPT2PreTrainedModel):
    def __init__(self, config, gpt, text_pos_emb, embeddings, norm, linear):
        super().__init__(config)
        self.transformer = gpt
        self.text_pos_embedding = text_pos_emb
        self.pos_embed  = text_pos_emb
        self.embeddings = embeddings
        self.lm_head = nn.Sequential(norm, linear)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.cached_emb = None

        self.cache_it   = 0

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def store_emb(self, emb):
        # store class embedding here
        self.cached_emb = emb

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):

        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        assert self.cached_emb is not None
        assert inputs_embeds is None  # Not supported by this inference model.
        assert labels is None  # Training not supported by this inference model.
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

       
        code_len = 1
        if input_ids.shape[1] != 1:
            start_emb  = self.embeddings(torch.tensor(1024)) + self.pos_embed(torch.tensor([[1024]]))
            emb = torch.cat([self.cached_emb.unsqueeze(0), start_emb.unsqueeze(0)], dim=1)
            
        else:
            emb = self.embeddings(input_ids)
            emb = emb + self.pos_embed.get_fixed_embedding(attention_mask.shape[1]-code_len, attention_mask.device)


        # Create embedding
        # mel_len = self.cached_mel_emb.shape[1]
        # if input_ids.shape[1] != 1:
        #     text_inputs = input_ids[:, mel_len:]
        #     text_emb = self.embeddings(text_inputs)
        #     text_emb = text_emb + self.text_pos_embedding(text_emb)
        #     if self.cached_mel_emb.shape[0] != text_emb.shape[0]:
        #         mel_emb = self.cached_mel_emb.repeat_interleave(text_emb.shape[0]//self.cached_mel_emb.shape[0], 0)
        #     else:
        #         mel_emb = self.cached_mel_emb
        #     emb = torch.cat([mel_emb, text_emb], dim=1)
        # else:
        #     emb = self.embeddings(input_ids)
        #     emb = emb + self.text_pos_embedding.get_fixed_embedding(attention_mask.shape[1]-mel_len, attention_mask.device)

        transformer_outputs = self.transformer(
            inputs_embeds=emb,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits,) + transformer_outputs[1:]

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past, beam_idx):
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PreTrainedModel.beam_search` or :meth:`~transformers.PreTrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )

if __name__=='__main__':

    datset_path = '/home/ubuntu/VQVAE/vq-vae-2-pytorch/exp1'

    dataset = LMDBDataset(datset_path)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=4, drop_last=True
    )
    batch = next(iter(loader))

    top, bottom, class_label = batch
    top          = einops.rearrange(top, 'b h w -> b (h w)')
    bottom       = einops.rearrange(bottom, 'b h w -> b (h w)') 

    top_index       = top.size(-1) 
    bottom_index    = bottom.size(-1) 



    model = GPTModelCIFAR(heads=16, layers=8)
    model_ckpt= '/home/ubuntu/VQVAE/vq-vae-2-pytorch/hf_ar/ckpt/model_1.pth'

    # print(model)
    model.load_state_dict(torch.load(model_ckpt))
    model.gpt.wte = model.embedding

    print("model loaded")

    inference_model = GPT2InferenceModel(
        model.gpt_config,
        model.gpt,
        model.wpe,
        model.embedding,
        model.final_norm,
        model.head
        )
    class_emb = model.class_embedding(class_label)

    class_emb = class_emb[0]
    inference_model.store_emb(class_emb)
    

    logits_processor = LogitsProcessorList()

    inputs = torch.ones((1, 2)).long()
    gen = inference_model.generate(inputs, bos_token_id=model.stop_token, pad_token_id=model.stop_token, eos_token_id=model.stop_token,
                                    max_length=82, logits_processor=logits_processor,
                                    num_return_sequences=1, temperature=0.65, top_p=0.9)

    print(gen)


    

