import torch
import torch.nn as nn
import numpy as np
import random
from layer import ModuleInjection, SuperScalableLinear

def set_glora(model, rank):
    layers = []
    for name, l in model.named_modules():
        if isinstance(l, nn.Linear):
            tokens = name.strip().split('.')
            layer = model
            for t in tokens[:-1]:
                if not t.isnumeric():
                    layer = getattr(layer, t)
                else:
                    layer = layer[int(t)]

            layers.append([layer, tokens[-1]])
    for parent_layer, last_token in layers:
        if not 'head' in last_token:
            setattr(parent_layer, last_token, ModuleInjection.make_scalable(getattr(parent_layer, last_token), rank))

def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def save(args, model):
    model.eval()
    model = model.cpu()
    trainable = {}
    for n, p in model.named_parameters():
        if any([x in n for x in ['A', 'B', 'C', 'D', 'E', 'head']]):
            trainable[n] = p.data
    torch.save(trainable, args.save_path + args.dataset + '.pt')

def load(args, vit):
    weights = torch.load(args.load_path + args.dataset + '.pt')
    loaded = 0
    for n, p in vit.named_parameters():
        if any([x in n for x in ['A', 'B', 'C', 'D', 'E', 'head']]):
            p.data = weights[n]
            loaded +=1
    print(f'successfully loaded {loaded} trained parameter tensors')
    return vit

def log(args, acc, ep):
    with open(args.save_path + args.dataset + '.log', 'a') as f:
        f.write(str(ep) + ' ' + str(acc) + '\n')