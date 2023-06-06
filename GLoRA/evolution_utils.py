import random
import os
import sys
import torch
from avalanche.evaluation.metrics.accuracy import Accuracy

class EvolutionSearcher(object):

    def __init__(self, args, device, model, choices, val_loader, output_dir):
        self.device = device
        self.model = model.to(device)
        self.args = args
        self.max_epochs = args.max_epochs
        self.select_num = args.select_num
        self.population_num = args.population_num
        self.m_prob = args.m_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num
        self.parameters_limits = args.param_limits 
        self.min_parameters_limits = args.min_param_limits 
        self.val_loader = val_loader
        self.output_dir = output_dir
        self.memory = []
        self.vis_dict = dict()
        self.keep_top_k = dict()
        self.keep_top_k[self.select_num] = []
        self.keep_top_k[50] = []
        self.epoch = 0
        self.candidates = []
        self.top_accuracies = []
        self.choices = choices
        self.cand_params = []

    def save_checkpoint(self):

        info = dict()
        info['top_accuracies'] = self.top_accuracies
        info['memory'] = self.memory
        info['candidates'] = self.candidates
        info['vis_dict'] = self.vis_dict
        info['keep_top_k'] = self.keep_top_k
        info['epoch'] = self.epoch
        checkpoint_path = os.path.join(self.output_dir, "checkpoint-{}.pth.tar".format(self.epoch))
        torch.save(info, checkpoint_path)
        print('save checkpoint to', checkpoint_path)

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_path):
            return False
        info = torch.load(self.checkpoint_path)
        self.memory = info['memory']
        self.candidates = info['candidates']
        self.vis_dict = info['vis_dict']
        self.keep_top_k = info['keep_top_k']
        self.epoch = info['epoch']

        print('load checkpoint from', self.checkpoint_path)
        return True

    def set_config(self, config):
        i = 0
        for name, l in self.model.named_modules():
            if isinstance(l, torch.nn.Linear) and name!='head':
                l.eval_config = config[i]
                i+=1

    def get_param_tensor(self, config, in_feature, out_feature, name):
        if 'A' in name or 'B' in name or 'C' in name:
            if 'C' in name:
                out_feature = in_feature
                in_feature = 1
            if 'LoRA' in config:
                try:
                    rank = int(config.split('_')[1])
                except:
                    rank = 16
                param = out_feature*rank + in_feature*rank
            elif 'vector' in config:
                param = out_feature
            elif 'constant' in config:
                param = 1
            elif 'none' in config:
                param = 0
            else:
                raise ValueError
        else:
            if 'vector' in config:
                param = out_feature
            elif 'constant' in config:
                param = 1
            elif 'none' in config:
                param = 0
            else:
                raise ValueError
        return param
    
    def get_param(self, configs):
        i = 0
        params = 0
        for n, l in self.model.named_modules():
            if isinstance(l, torch.nn.Linear) and n != 'head':
                out_channel = l.out_features
                in_channel = l.in_features
                for sup_tnsr in ['A', 'B', 'C', 'D', 'E']:
                    params += self.get_param_tensor(configs[i][sup_tnsr], in_channel, out_channel, sup_tnsr)
                i+=1
        return params

    def is_legal(self, cand):
        assert isinstance(cand, tuple)
        if str(cand) not in self.vis_dict:
            self.vis_dict[str(cand)] = {}
        info = self.vis_dict[str(cand)]
        if 'visited' in info:
            return False
        n_parameters = self.get_param(configs=cand)
        info['params'] =  n_parameters / 10.**6 

        if info['params'] > self.parameters_limits:
            print('parameters limit exceed')
            sys.stdout.flush()
            return False

        if info['params'] < self.min_parameters_limits:
            print('under minimum parameters limit')
            return False
        
        eval_acc = self.evaluate(config=cand)
        info['acc'] = eval_acc
        info['visited'] = True

        return True

    def evaluate(self, config):
        self.model.eval()
        self.set_config(config)
        acc = Accuracy()
        for batch in self.val_loader:
            x, y = batch[0].cuda(), batch[1].cuda()
            out = self.model(x).data
            acc.update(out.argmax(dim=1).view(-1), y, 1)

        return acc.result()[1]
    
    def update_top_k(self, candidates, *, k, key, reverse=True):
        assert k in self.keep_top_k
        print('select ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def stack_random_cand(self, random_func, *, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                if str(cand) not in self.vis_dict:
                    self.vis_dict[str(cand)] = {}
                info = self.vis_dict[str(cand)]
            for cand in cands:
                yield cand

    def get_random_cand(self):

        cand_tuple = list()
        depth = 48 ## 12 (depth) X 4 (layers)
        for i in range(depth):
            cand_tuple.append({'A':random.choice(self.choices['A']),
                               'B':random.choice(self.choices['B']),
                               'C':random.choice(self.choices['C']),
                               'D':random.choice(self.choices['D']),
                               'E':random.choice(self.choices['E'])})

        return tuple(cand_tuple)

    def get_random(self, num):
        print('random select ........')
        cand_iter = self.stack_random_cand(self.get_random_cand)
        while len(self.candidates) < num:
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            print('random {}/{}'.format(len(self.candidates), num))
        print('random_num = {}'.format(len(self.candidates)))

    def get_mutation(self, k, mutation_num, m_prob):
        assert k in self.keep_top_k
        print('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = list(random.choice(self.keep_top_k[k]))
            final = list()
            for i in range(len(cand)):
                final_layer = dict()
                for key in ['A', 'B', 'C', 'D', 'E']:
                    random_s = random.random()
                    if random_s < m_prob:
                        final_layer[key] = random.choice(self.choices[key])
                    else:
                        final_layer[key] = cand[i][key]
                final.append(final_layer)
            return tuple(final)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print('mutation {}/{}'.format(len(res), mutation_num))

        print('mutation_num = {}'.format(len(res)))
        return res

    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        print('crossover ......')
        res = []
        iter = 0
        max_iters = 10 * crossover_num

        def random_func():
            cand_1 = list(random.choice(self.keep_top_k[k]))
            cand_2 = list(random.choice(self.keep_top_k[k]))
            final = list()
            for i in range(len(cand_1)):
                final_layer = dict()
                for key in ['A', 'B', 'C', 'D', 'E']:
                    final_layer[key] = random.choice([cand_1[i][key], cand_2[i][key]])
                final.append(final_layer)
            return tuple(final)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print('crossover {}/{}'.format(len(res), crossover_num))

        print('crossover_num = {}'.format(len(res)))
        return res

    def search(self):
        print(
            'population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
                self.population_num, self.select_num, self.mutation_num, self.crossover_num,
                self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))


        self.get_random(self.population_num)

        while self.epoch < self.max_epochs:
            print('epoch = {}'.format(self.epoch))

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)
            
            #updata top10
            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[str(x)]['acc'])
            #updata top50
            self.update_top_k(
                self.candidates, k=50, key=lambda x: self.vis_dict[str(x)]['acc'])

            print('epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[50])))
            tmp_accuracy = []
            for i, cand in enumerate(self.keep_top_k[50]):
                print('No.{} Top-1 val acc = {}, params = {}'.format(
                    i + 1, self.vis_dict[str(cand)]['acc'], self.vis_dict[str(cand)]['params']))   
                sys.stdout.flush()
                tmp_accuracy.append(self.vis_dict[str(cand)]['acc'])
            self.top_accuracies.append(tmp_accuracy)

            mutation = self.get_mutation(
                self.select_num, self.mutation_num, self.m_prob)
            crossover = self.get_crossover(self.select_num, self.crossover_num)

            self.candidates = mutation + crossover

            self.get_random(self.population_num)

            self.epoch += 1

            self.save_checkpoint()