from pickletools import optimize
import numpy as np
import einops


import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from dataset import LMDBDataset
from hf_ar.model import GPTModelCIFAR




################### Params ##################

datset_path = '/home/ubuntu/VQVAE/vq-vae-2-pytorch/exp1'
ckpt_path   = '/home/ubuntu/VQVAE/vq-vae-2-pytorch/hf_ar/ckpt/'
bs          = 256
num_classes = 10
epochs      = 100000
save_interval = 2000
log_interval = 500

lr           = 5e-5
##############################################



model = GPTModelCIFAR(heads=16, layers=8)
model = model.cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr= lr,)
criterion = nn.CrossEntropyLoss()

dataset = LMDBDataset(datset_path)
loader = DataLoader(
    dataset, batch_size=bs, shuffle=True, num_workers=4, drop_last=True
)


total_steps = 0
for epoch in range(epochs):

    for steps, (top, bottom, class_label) in enumerate(loader):
        
        optimizer.zero_grad()
        top = top 
        bottom = bottom 
        class_label = class_label.cuda()

        top          = einops.rearrange(top, 'b h w -> b (h w)')
        bottom       = einops.rearrange(bottom, 'b h w -> b (h w)') 

        top_index       = top.size(-1) 
        bottom_index    = bottom.size(-1) 

        emb = torch.cat((top,bottom), dim = -1)
        emb = emb.cuda()
        
        logits, targets = model(class_label, emb)
        loss   = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        if(total_steps % log_interval == 0):
            print("Step Loss: ", loss.item(), f"Steps: {total_steps}", f"Epoch: {epoch}",)

        if(total_steps % save_interval == 0):
            print("\nnew model saved\n")
            torch.save(model.state_dict(), f'{ckpt_path}model_{total_steps//save_interval}.pth')

        total_steps += 1