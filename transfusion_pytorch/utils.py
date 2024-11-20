#!/usr/bin/env python3
import torch
import argparse


def TransfusionArgParser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
     
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=384)
    parser.add_argument('--tokenizer', type=str, default="meta-llama/Llama-2-7b-hf")
    
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
        
         
def scope_kwargs(**kwargs):
    out = {}
    for k,v in kwargs.items():
        namespace = out
        scopes = k.split('.')
        for i, scope in enumerate(scopes):
            namespace = namespace.setdefault(scope, {} if i < len(scopes)-1 else v)  
    return out


def print_system_info():
    print(torch.__config__.show())
    print(f'PyTorch version: {torch.__version__}')
    
    if torch.cuda.is_available():
        print(f'CUDA version:    {torch.version.cuda} ({torch.cuda.get_device_name()})')
        print(f'cuDNN version:   {torch.backends.cudnn.version()}\n')


def print_model_info(model):
    try:
        import torchinfo
        USE_TORCHINFO=True
    except ImportError as error:
        print(f"Disabling use of torchinfo ({error})\npip install torchinfo for additional statistics")
        USE_TORCHINFO=False
   
    print(model)
    
    if USE_TORCHINFO:
        torchinfo.summary(model)
    else:
        print('Total params:', sum(p.numel() for p in model.parameters()))
        print('Train params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    
