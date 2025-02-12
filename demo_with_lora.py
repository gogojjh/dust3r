#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# dust3r gradio demo executable
# --------------------------------------------------------

# python demo_with_lora.py --weights /Rocket_ssd/image_matching_model_weights/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth

import os
import torch
import tempfile

from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.demo import get_args_parser, main_demo, set_print_with_timestamp

import matplotlib.pyplot as pl
pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    set_print_with_timestamp()

    if args.tmp_dir is not None:
        tmp_path = args.tmp_dir
        os.makedirs(tmp_path, exist_ok=True)
        tempfile.tempdir = tmp_path

    if args.server_name is not None:
        server_name = args.server_name
    else:
        server_name = '0.0.0.0' if args.local_network else '127.0.0.1'

    if args.weights is not None:
        weights_path = args.weights
    else:
        weights_path = "naver/" + args.model_name
    model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(args.device)

    USE_LORA = True
    if USE_LORA:
        from torch import nn
        from dust3r.lora import LoraLayer, inject_lora

        # Traverse all lora layer
        for name,layer in model.named_modules():
            name_cols = name.split('.')
            filter_names = ['qkv']
            if any(n in name_cols for n in filter_names) and isinstance(layer, nn.Linear):
                inject_lora(model, name, layer)

        try:
            restore_lora_state = torch.load('/Rocket_ssd/image_matching_model_weights/dust3r_demo_512dpt_lora/lora.pt')
            restore_lora_state = {k.replace('module.', ''): v for k, v in restore_lora_state.items()}
            model.load_state_dict(restore_lora_state, strict=False)
            print('Finish loading LoRA weights')

            # print("Keys in restore_lora_state:", restore_lora_state.keys())
            # print("Model's state_dict keys:", model.state_dict().keys())

            # for name, param in model.named_parameters():
            #     if 'lora_a' in name or 'lora_b' in name:
            #         print(f"{name}: mean={param.mean().item()}, max={param.max().item()}")            
            # exit()
        except:
            pass 

        model = model.to(args.device)

        # Add lora weights into the model weight as the linear layer
        for name, layer in model.named_modules():
            name_cols = name.split('.')
            if isinstance(layer, LoraLayer):
                children = name_cols[:-1]
                cur_layer = model 
                for child in children:
                    cur_layer = getattr(cur_layer,child)  
                lora_weight = (layer.lora_a @ layer.lora_b) * layer.alpha / layer.r
                print(lora_weight)
                layer.raw_linear.weight = nn.Parameter(layer.raw_linear.weight.add(lora_weight.T)).to(args.device)
                setattr(cur_layer, name_cols[-1], layer.raw_linear)

    print('Number of Model Parameters: ', sum(p.numel() for p in model.parameters()))
    print('Number of LoRA  Parameters: ', sum(param.numel() for param in restore_lora_state.values()))

    # dust3r will write the 3D model inside tmpdirname
    with tempfile.TemporaryDirectory(suffix='dust3r_gradio_demo') as tmpdirname:
        if not args.silent:
            print('Outputing stuff in', tmpdirname)
        main_demo(tmpdirname, model, args.device, args.image_size, server_name, args.server_port, silent=args.silent)
