import torch
from torch import nn, randint, randn
from transfusion_pytorch import Transfusion, print_modality_sample

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

mock_encoder = nn.Conv2d(3, 384, 3, padding = 1).to(device)
mock_decoder = nn.Conv2d(384, 3, 3, padding = 1).to(device)

model = Transfusion(
    num_text_tokens = 12,
    dim_latent = 384,
    channel_first_latent = True,
    modality_default_shape = (4, 4),
    modality_encoder = mock_encoder,
    modality_decoder = mock_decoder,
    transformer = dict(
        dim = 512,
        depth = 8
    )
).to(device)

print(model, model.device)

print('Total params:', sum(p.numel() for p in model.parameters()))
print('Train params:', sum(p.numel() for p in model.parameters() if p.requires_grad))

text_and_images = [
    [
        randint(0, 12, (16,)),  # 16 text tokens
        randn(3, 8, 8),         # (8 x 8) 3 channeled image
        randint(0, 12, (8,)),   # 8 text tokens
        randn(3, 7, 7)          # (7 x 7) 3 channeled image
    ],
    [
        randint(0, 12, (16,)),  # 16 text tokens
        randn(3, 8, 5),         # (8 x 5) 3 channeled image
        randint(0, 12, (5,)),   # 5 text tokens
        randn(3, 2, 16),        # (2 x 16) 3 channeled image
        randint(0, 12, (9,))    # 9 text tokens
    ]
]

loss = model(text_and_images)

loss.backward()

# after much training, generate multimodal sample
model.eval()

with torch.no_grad():
    sample = model.sample(max_length=64)

print_modality_sample(sample)

