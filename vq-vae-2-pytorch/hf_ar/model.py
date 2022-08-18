import functools
from transformers import GPT2Config, GPT2Model
import torch
import torch.nn.functional as F

import torch.nn as nn

from torch.utils.data import DataLoader
import einops

def null_position_embeddings(range, dim):
    return torch.zeros((range.shape[0], range.shape[1], dim), device=range.device)

class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=.02):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        # Initializing this way is standard for GPT-2
        self.emb.weight.data.normal_(mean=0.0, std=init)

    def forward(self, x):
        sl = x.shape[1]
        return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, ind, dev):
        return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)


class GPTModelCIFAR(nn.Module):
    def __init__(
        self,
        codebook_len_t = 16,
        codebook_len_b = 64,
        codebook_size_t = 512,
        codebook_size_b = 512,

        model_dim = 1024,
        layers = 16,
        heads = 16,
        num_classes   = 10, #CIFAR 10
        checkpointing = False,
    ):

        self.total_codebook_size = codebook_size_t + codebook_size_b
        self.start_token = self.total_codebook_size
        self.stop_token  = self.total_codebook_size + 1 

        
        super(GPTModelCIFAR, self).__init__()

        self.gpt_config = GPT2Config(vocab_size=256,  # Unused.
                                    n_positions=codebook_len_t+codebook_len_b + 4,
                                    n_ctx=codebook_len_t+codebook_len_b + 4,
                                    n_embd=model_dim,
                                    n_layer=layers,
                                    n_head=heads,
                                    gradient_checkpointing=checkpointing,
                                    use_cache=not checkpointing)

        self.gpt = GPT2Model(self.gpt_config)
        # Override the built in positional embeddings
        del self.gpt.wpe
        self.gpt.wpe = functools.partial(null_position_embeddings, dim=model_dim)
        # Built-in token embeddings are unused.
        del self.gpt.wte

        self.wpe  = LearnedPositionEmbeddings(seq_len=codebook_len_t + codebook_len_b + 50, model_dim=model_dim)

        self.embedding     = nn.Embedding(codebook_size_t + 514, model_dim)
        self.class_embedding = nn.Embedding(num_classes, model_dim)

        self.head  = nn.Linear(model_dim, codebook_size_t + 514) # add 2 for each start and stop tokens
        self.final_norm = nn.LayerNorm(model_dim)

    

    def forward(self, img_class, emb):
        
        #flatten

        #add start stop token
        #might not be necessary


        emb       = F.pad(emb, (1,0), "constant", self.start_token)
        emb       = F.pad(emb, (0,1), "constant", self.stop_token)

        targets   = emb[:, 1:]
        orig_emb  = emb

        class_emb = self.class_embedding(img_class)
        emb   = self.embedding(emb) + self.wpe(emb)
        emb     = torch.cat((class_emb, emb), dim= 1)
        gpt_out = self.gpt(inputs_embeds=emb, return_dict=True, output_attentions=False)

        # ignore class embed and stop token
        enc = gpt_out.last_hidden_state[:, 1:-1]
        enc = self.final_norm(enc)

        
        logits = self.head(enc)
        logits = logits.permute(0,2,1)

        # targets [ 1, 2, 3, 4, stop]
        #         [ s, 1, 2, 3, 4]

        return logits, targets




if __name__=="__main__":

    criterion1  = nn.CrossEntropyLoss()
    criterion2  = nn.CrossEntropyLoss()

    dataset    = GPT2DatasetSupervised()
    dataloader = DataLoader(dataset, batch_size= 4)
    model      = GPTModelCIFAR()

    for batch in dataloader:
        speaker_l, text_l, mask_l, latents_l = batch
        break
    
    
    target = torch.cat((text_l, latents_l), dim= 1)


    text_target = target[:, 1:text_l.shape[1]]
    text_target = F.pad(text_target, (0,1), "constant", dataset.mel_start_token)

    mel_target = target[:, -latents_l.shape[1]:]
    mel_target = mel_target[:, 1:-1]

    text_logits, mel_logits    = model(speaker_l, text_l, mask_l, latents_l)


    loss1 = criterion1(text_logits, text_target)
    loss2 = criterion2(mel_logits, mel_target)


    x = 0
    
