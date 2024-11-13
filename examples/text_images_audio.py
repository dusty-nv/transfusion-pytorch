#!/usr/bin/env python3
import torch

from torch import randint, randn
from transfusion_pytorch import Transfusion, print_modality_sample

model = Transfusion(
    num_text_tokens = 256,
    dim_latent = (384, 192),                 # specify multiple latent dimensions
    modality_default_shape = ((4,), (2,)),   # default shapes for first and second modality
    transformer = dict(
        dim = 512,
        depth = 8
    )
).to(device='cuda:0' if torch.cuda.is_available() else 'cpu')

print(model)

print('Total params:', sum(p.numel() for p in model.parameters()))
print('Train params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
print('Transformer params:', sum(p.numel() for p in model.parameters()))

# then for the Tensors of type float, you can pass a tuple[int, Tensor] and specify the modality index in the first position

# any torch.long is text, torch.float is modalities

text_images_and_audio = [
    [randint(0, 256, (16,)), (0, randn(4, 384)), randint(0, 256, (8,)), (1, randn(6, 192))],
    [randint(0, 256, (16,)), randn(7, 384), randint(0, 256, (5,)), (1, randn(2, 192)), randint(0, 256, (9,))]
]

loss = model(text_images_and_audio)

loss.backward()

# after much training, generate multimodal sample
model.eval()

with torch.no_grad():
    sample = model.sample(max_length=64)

print_modality_sample(sample)

