#!/usr/bin/env python3
import argparse
import torch

from tqdm import tqdm
from pprint import pprint

from torch import randint, randn
from torch.profiler import profile as Profiler, ProfilerActivity

from transfusion_pytorch import Transfusion, print_modality_sample

from .utils import TransfusionArgParser, print_model_info, print_system_info, scope_kwargs
    
    
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

    print_model_info(model)
    
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
    
    if profile:
        profiler_activities = [ProfilerActivity.CUDA] # ProfilerActivity.CPU, 
        profiler = Profiler(activities=profiler_activities, record_shapes=True, profile_memory=True)
        profiler.start()

    with tqdm(total=train_steps) as pbar:
        for step in range(train_steps):
            loss = model(batch)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            pbar.set_description(f'train loss={loss.item():.3f}')
            pbar.update(1)
            
            if profile:
                profiler.step()
     
    print_modality_sample(model.sample(max_length=max_gen_length))
    
    if profile:
        profiler.stop()
        prof_averages = profiler.key_averages()
        print(prof_averages.table(sort_by='cuda_time_total', row_limit=25, top_level_events_only=True))
        print(prof_averages.table(sort_by='cuda_memory_usage', row_limit=25, top_level_events_only=True))

                  
if __name__ == '__main__': 
    print_system_info()
    
    parser = TransfusionArgParser()

    parser.add_argument('--train_steps', type=int, default=16, help='the number of training batches to run')
    parser.add_argument('--text_length', type=int, default=16, help='the number of tokens in each text message')
    parser.add_argument('--max_gen_length', type=int, default=32, help='the number of tokens in each text message')
    parser.add_argument('--modality_pattern', type=str, default=[], nargs='+', help="the modality pattern to follow when constructing contexts (list of 'text', 'image')")
    parser.add_argument('--profile', action='store_true', help='enable layer-level profiling and memory usage in PyTorch')
    
    args = scope_kwargs(**vars(parser.parse_args()))
    pprint(args, indent=2)
    benchmark(**args)

