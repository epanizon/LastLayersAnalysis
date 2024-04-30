import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict
from useless_layers.extract_activations import extract_activations
from transformers import PreTrainedModel
import sys
import pickle


def get_embdims(model, dataloader, target_layers):
    embdims = defaultdict(lambda: None)
    dtypes = defaultdict(lambda: None)

    def get_hook(name, embdims):
        def hook_fn(module, input, output):
            if type(output) is tuple:
                output = output[0]
            embdims[name] = output.shape[-1]
            dtypes[name] = output.dtype

        return hook_fn

    handles = {}
    for name, module in model.named_modules():
        if name in target_layers:
            handles[name] = module.register_forward_hook(get_hook(name, embdims))

    batch = next(iter(dataloader))
    _ = model(batch["input_ids"].to("cuda"))

    for name, module in model.named_modules():
        if name in target_layers:
            handles[name].remove()

    assert len(embdims) == len(target_layers)
    return embdims, dtypes


@torch.inference_mode()
def save_act_dict(
    model: PreTrainedModel,   #check
    model_name: str,          #check
    dataloader: DataLoader,   #check
    target_layers: dict,      #check
    nsamples,                 #check
    dirpath=".",              #check
    filename="",              #check
    use_last_token=False,     #need to do all tokens, but for now ok.
    print_every=1,          
):
    dirpath = str(dirpath).lower()
    os.makedirs(dirpath, exist_ok=True)
    if use_last_token:
        filename = f"_{filename}_target"
    else:
        filename = f"_{filename}_mean"
    target_layer_names = list(target_layers.values())
    target_layer_labels = list(target_layers.keys())
    model = model.eval()
    start = time.time()
    print("layer_to_extract: ", target_layer_labels)
    embdims, dtypes = get_embdims(model, dataloader, target_layer_names)
    extr_act = extract_activations(  ## MOMOMO
        model,
        model_name,
        dataloader,
        target_layer_names,
        embdims,
        dtypes,
        nsamples,
        use_last_token=use_last_token,
        print_every=print_every,
    )
    extr_act.extract(dataloader)
    print(f"num_tokens: {extr_act.tot_tokens/10**3}k")
    print((time.time() - start) / 3600, "hours")
    act_dict = extr_act.hidden_states

    for layer in act_dict.keys():
        torch.save(act_dict[layer], dirpath + '/' + layer + '.pt')
