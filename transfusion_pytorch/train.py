#!/usr/bin/env python3
import os
import json
import pprint
import argparse

import torch
import torchvision
import torchvision.transforms.functional as F

from transformers import AutoTokenizer
from torchvision.transforms import v2 as tv2
from transfusion_pytorch import Transfusion, print_modality_sample

from .utils import TransfusionArgParser, print_model_info, print_system_info, scope_kwargs
   

class PrepareData:
    def __init__(self, tokenizer, image_size=384, class_labels=None, device=None):
        self.device = device
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.class_labels = None
        
        if class_labels:
            with open(class_labels) as file:
                self.class_labels = json.load(file)

        self.image_transform = tv2.Compose([
            tv2.ToTensor(),
            tv2.Resize((image_size, image_size)),
            tv2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def image(self, data):
        return self.image_transform(data).to(self.device)
        
    def text(self, data):
        if self.class_labels:
            txt = list(self.class_labels.keys())[data]
        else:
            txt = str(data)
            
        return self.tokenizer.encode(txt, add_special_tokens=False, return_tensors='pt').to(self.device)
        
    def collate(self, data):
        return [
            [x[1].squeeze(), x[0].squeeze()] for x in data
        ]


def train(
    dataset=dict(
        name='Flowers102',
        path='./datasets/Flowers102',
        split='train',
    ),
    tokenizer='meta-llama/Llama-2-7b-hf',
    trainer=dict(
        epochs=1,
        steps=None,
        lr=3e-4,
    ),
    batch_size=1, 
    image_size=384, 
    modality_pattern=[], 
    dim_latent=384, 
    **kwargs
):
    if not modality_pattern:
        modality_pattern = ['text', 'image'] #, 'text', 'image']

    if isinstance(dim_latent, int):
        dim_latent = (dim_latent,)
        
    if not isinstance(dim_latent, tuple):
        dim_latent = tuple(dim_latent)

    device ='cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # load tokenizer
    token_cfg = tokenizer
    tokenizer = AutoTokenizer.from_pretrained(token_cfg)

    print(f"Tokenizer {token_cfg} vocabulary length: {len(tokenizer)}")
    kwargs['num_text_tokens'] = len(tokenizer)
    
    # create model
    encoder = torch.nn.Conv2d(3, image_size, 3, padding = 1).to(device)
    decoder = torch.nn.Conv2d(image_size, 3, 3, padding = 1).to(device)

    model = Transfusion(
        dim_latent=dim_latent, 
        channel_first_latent = True,
        modality_default_shape=tuple([4]*len(dim_latent)),
        modality_encoder=encoder,
        modality_decoder=decoder,
        **kwargs
    ).to(device)
    
    print_model_info(model)
    

    # load dataset
    data_prep = PrepareData(
        tokenizer=tokenizer,
        image_size=image_size,
        class_labels=f"{os.path.dirname(__file__)}/../examples/flowers102.json",
        device=device,
    )
    
    dataset = getattr(torchvision.datasets, dataset['name'])(
        dataset['path'], split=dataset['split'], download=True,
        transform=data_prep.image, target_transform=data_prep.text,
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        collate_fn=data_prep.collate,
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=trainer['lr'])

    for epoch in range(trainer['epochs']):
        for i, batch in enumerate(dataloader):
            #batch_desc = [f'{x.shape} {x.dtype} {x.device}' for x in y for y in batch]
            print(f"epoch {epoch} [{i}/{len(dataloader)}]")#  {batch_desc}")
            #print(batch)
            
            loss = model(batch)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

     
            #print_modality_sample(model.sample(max_length=max_gen_length))


if __name__ == '__main__': 
    print_system_info()
    
    parser = TransfusionArgParser()

    parser.add_argument('--dataset.name', type=str, default='Flowers102')
    parser.add_argument('--dataset.path', type=str, default='/data/datasets/torchvision/Flowers102')
    parser.add_argument('--dataset.split', type=str, default='train')
    
    parser.add_argument('--trainer.epochs', type=int, default=1, help='the number of training epochs to run over the dataset')
    parser.add_argument('--trainer.steps', type=int, default=None, help='the number of training batches to run')
    parser.add_argument('--trainer.lr', type=float, default=3e-4, help='optimizer learning rate')

    parser.add_argument('--modality_pattern', type=str, default=[], nargs='+', help="the modality pattern to follow when constructing contexts (list of 'text', 'image')")

    args = scope_kwargs(**vars(parser.parse_args()))
    pprint.pprint(args, indent=2)
    train(**args)

