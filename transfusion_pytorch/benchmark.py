#!/usr/bin/env python3
import argparse

import torch
import torchinfo

from tqdm import tqdm
from pprint import pprint

from torch import randint, randn
from torch.profiler import profile as profiler, record_function, ProfilerActivity

from transfusion_pytorch import Transfusion, print_modality_sample


def benchmark(
    batch_size=1, 
    image_size=16, 
    text_length=16, 
    max_gen_length=64,
    modality_pattern=[], 
    dim_latent=384, 
    train_steps=128, 
    profile=False,
    **kwargs
):
    if not modality_pattern:
        modality_pattern = ['text', 'image'] #, 'text', 'image']
    
    if isinstance(dim_latent, int):
        dim_latent = (dim_latent,)
        
    if not isinstance(dim_latent, tuple):
        dim_latent = tuple(dim_latent)

    device ='cuda:0' if torch.cuda.is_available() else 'cpu'
    
    model = Transfusion(
        dim_latent=dim_latent, 
        modality_default_shape=tuple([2]*len(dim_latent)),
        **kwargs).to(device)
    
    print(model)
    torchinfo.summary(model)
    #print('Total params:', sum(p.numel() for p in model.parameters()))
    #print('Train params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    vocab = kwargs.get('num_text_tokens', 256)
    batch = []
    
    for i in range(batch_size):
        context = []
        for pattern in modality_pattern:
            pattern = pattern.strip().lower()
            if pattern == 'text':
                context += [randint(0, vocab, (text_length,))]
            elif pattern == 'image':  # TODO adapt to correspond with dim_latent array
                context += [randn(0, image_size, dim_latent[0])]
        batch.append(context)
                 
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    profiler_activities = [ProfilerActivity.CUDA] # ProfilerActivity.CPU, 
    
    with profiler(activities=profiler_activities, record_shapes=True, profile_memory=True) as prof:
        with record_function('train'):
            with tqdm(total=train_steps) as pbar:
                for step in range(train_steps):
                    loss = model(batch)
                    loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()

                    pbar.set_description(f'train loss={loss.item():.3f}')
                    pbar.update(1)
                    prof.step()
        
        with record_function('sample'):       
            print_modality_sample(model.sample(max_length=max_gen_length))

    prof_averages = prof.key_averages()
    
    print(prof_averages.table(sort_by='cuda_time_total', row_limit=25, top_level_events_only=True))
    print(prof_averages.table(sort_by='cuda_memory_usage', row_limit=25, top_level_events_only=True))
    
def scope_kwargs(**kwargs):
    out = {}
    for k,v in kwargs.items():
        namespace = out
        scopes = k.split('.')
        for i, scope in enumerate(scopes):
            namespace = namespace.setdefault(scope, {} if i < len(scopes)-1 else v)  
    return out

def TransfusionArgParser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
     
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=8)

    parser.add_argument('--num_text_tokens', '--vocab_size', type=int, default=256, help='size of the tokenizer vocabulary (number of token IDs)')
    parser.add_argument('--dim_latent', type=int, nargs='+', default=[384], help='modality latency dimensions (can specify multiple)')

    parser.add_argument('--transformer.dim', type=int, default=512)
    parser.add_argument('--transformer.depth', type=int, default=8)
    parser.add_argument('--transformer.heads', type=int, default=8)
    parser.add_argument('--transformer.dim_head', type=int, default=64)
    parser.add_argument('--transformer.dropout', type=float, default=0.0)
    parser.add_argument('--transformer.ff_expansion_factor', type=int, default=4)
    parser.add_argument('--transformer.unet_skips', type=bool, default=True)
    parser.add_argument('--transformer.use_flex_attn', type=bool, default=False)

    return parser
         
                  
if __name__ == '__main__': 
    print(torch.__config__.show())
    print(f'PyTorch version: {torch.__version__}')
    
    if torch.cuda.is_available():
        print(f'CUDA version:    {torch.version.cuda} ({torch.cuda.get_device_name()})')
        print(f'cuDNN version:   {torch.backends.cudnn.version()}\n')
        
    parser = TransfusionArgParser()

    parser.add_argument('--train_steps', type=int, default=16, help='the number of training batches to run')
    parser.add_argument('--text_length', type=int, default=16, help='the number of tokens in each text message')
    parser.add_argument('--max_gen_length', type=int, default=32, help='the number of tokens in each text message')
    parser.add_argument('--modality_pattern', type=str, default=[], nargs='+', help="the modality pattern to follow when constructing contexts (list of 'text', 'image')")
    parser.add_argument('--profile', action='store_true', help='enable layer-level profiling and memory usage in PyTorch')
    
    args = scope_kwargs(**vars(parser.parse_args()))
    pprint(args, indent=2)
    benchmark(**args)

