#!/usr/bin/env python3
import torch

from transfusion_pytorch import Transfusion, print_modality_sample

model = Transfusion(
    num_text_tokens = 256,
    dim_latent = 384,
    transformer = dict(
        dim = 512,
        depth = 8,
    )
).to(device='cuda:0' if torch.cuda.is_available() else 'cpu')

print(model)

print('Total params:', sum(p.numel() for p in model.parameters()))
print('Train params:', sum(p.numel() for p in model.parameters() if p.requires_grad))

text = torch.randint(0, 256, (2, 1024)).cuda() # batch size 2

loss = model(text)
loss.backward()

# after much training, next token prediction
model.eval()

with torch.no_grad():
    samples = model.generate_text_only(text[:, :1], 64)

print_modality_sample(samples)
