import torch
from timm.models import create_model
from argparse import ArgumentParser
from data import *
from pathlib import Path
from evolution_utils import EvolutionSearcher
from utils import set_seed, set_glora, load

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--model', type=str, default='vit_base_patch16_224_in21k')
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--save_path', type=str, default='models/temp/')
    parser.add_argument('--load_path', type=str, default='models/temp/')
    parser.add_argument('--max-epochs', type=int, default=20)
    parser.add_argument('--select-num', type=int, default=10)
    parser.add_argument('--population-num', type=int, default=50)
    parser.add_argument('--m_prob', type=float, default=0.2)
    parser.add_argument('--crossover-num', type=int, default=25)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--mutation-num', type=int, default=25)
    parser.add_argument('--param-limits', type=float, default=1.00)
    parser.add_argument('--min-param-limits', type=float, default=0)
    parser.add_argument('--rank', type=int, default=4)
    args = parser.parse_args()
    seed = args.seed
    set_seed(seed)
    device = torch.device('cuda:0')
    name = args.dataset
    args.best_acc = 0
    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    vit = create_model(args.model, checkpoint_path='./ViT-B_16.npz', drop_path_rate=0.1)
    set_glora(vit, args.rank)
    train_dl, test_dl = get_data(name)

    vit.reset_classifier(get_classes_num(name))
    vit = load(args, vit)
    for n, p in vit.named_parameters():
        p.requires_grad = False

    choices = dict()
    choices['A'] = [f'LoRA_{args.rank}', 'vector', 'constant', 'none']
    choices['B'] = choices['A']
    choices['C'] = [f'LoRA_{args.rank}', 'vector', 'none']
    choices['D'] = ['constant', 'none', 'vector']
    choices['E'] = choices['D']
    searcher = EvolutionSearcher(args, device, vit, choices, test_dl, args.save_path)
    searcher.search()