#!/usr/bin/env python3
import torch

from torch import randint, randn
from transfusion_pytorch import Transfusion, print_modality_sample

model = Transfusion(
    num_text_tokens = 256,
    dim_latent = 384,
    modality_default_shape = (4,),  # fallback, in the case the language model did not produce a valid modality shape
    transformer = dict(
        dim = 512,
        depth = 8
    )
).to(device='cuda:0' if torch.cuda.is_available() else 'cpu')

print(model)

print('Total params:', sum(p.numel() for p in model.parameters()))
print('Train params:', sum(p.numel() for p in model.parameters() if p.requires_grad))

# any torch.long is text, torch.float is modalities

text_and_images = [
    [randint(0, 256, (16,)), randn(4, 384), randint(0, 256, (8,)), randn(6, 384)],
    [randint(0, 256, (16,)), randn(7, 384), randint(0, 256, (5,)), randn(2, 384), randint(0, 256, (9,))]
]

loss = model(text_and_images)

loss.backward()

# after much training, generate multimodal sample
model.eval()

with torch.no_grad():
    sample = model.sample(max_length=64)

print_modality_sample(sample)

