import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class SuperScalableLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, rank):
        super(SuperScalableLinear, self).__init__(in_features=in_features, out_features=out_features)
        config_A_B = [f'LoRA_{rank}', 'vector', 'constant', 'none']
        config_C = [f'LoRA_{rank}', 'vector', 'none']
        config_D_E = ['constant', 'none', 'vector']
        self.configs = []
        for A in config_A_B:
            for B in config_A_B:
                for C in config_C:
                    for D in config_D_E:
                        for E in config_D_E:
                            config = {'A':A,'B':B,'C':C,'D':D,'E':E}
                            self.configs.append(config)

        self.Ad, self.Au = self.make_param((out_features, in_features), f'LoRA_{rank}')
        self.Bd, self.Bu = self.make_param((out_features, in_features), f'LoRA_{rank}')
        self.Cd, self.Cu = self.make_param((in_features, 1), f'LoRA_{rank}')
        self.D = nn.Parameter(torch.zeros(out_features))
        self.E = nn.Parameter(torch.zeros(out_features))
        self.eval_config = None
        nn.init.xavier_uniform_(self.Au)
        nn.init.xavier_uniform_(self.Bu)
        nn.init.xavier_uniform_(self.Cu)
    
    def prepare_path(self, config, Xd, Xu=None):
        if Xu is not None:
            if 'LoRA' in config:
                rank = int(config.split('_')[1])
                X = torch.matmul(Xd[:,:rank], Xu[:rank, :])
            elif 'vector' in config:
                X = Xd[:,0].unsqueeze(1)
            elif 'constant' in config:
                X = Xd[0,0]
            elif 'none' in config:
                X = torch.zeros(Xd.shape[0], Xu.shape[1]).cuda()
            else:
                raise ValueError
        else:
            if 'vector' in config:
                X = Xd
            elif 'constant' in config:
                X = Xd[0]
            elif 'none' in config:
                X = torch.zeros(1).cuda()
            else:
                raise ValueError
        return X
    
    def make_param(self, shape, config=None):
        if 'LoRA' in config:
            out_feature = shape[0]
            in_feature = shape[1]
            try:
                rank = int(config.split('_')[1])
            except:
                rank = 4
            return nn.Parameter(torch.zeros(out_feature, rank)), nn.Parameter(torch.zeros(rank, in_feature))
        return nn.Parameter(torch.zeros(*shape))
        
    def forward(self, input):
        if self.eval_config is not None:
            path_config = self.eval_config
        else:
            path_config = random.choice(self.configs)
        A = self.prepare_path(path_config['A'], self.Ad, self.Au)
        B = self.prepare_path(path_config['B'], self.Bd, self.Bu)
        C = self.prepare_path(path_config['C'], self.Cd, self.Cu)
        D = self.prepare_path(path_config['D'], self.D)
        E = self.prepare_path(path_config['E'], self.E)
        optimal_weight = self.weight + self.weight*A + B
        if torch.is_tensor(self.bias):
            optimal_bias = self.bias + self.bias*D + E
        else:
            optimal_bias = E
        optimal_prompt = torch.matmul(self.weight, C).squeeze()
        return F.linear(input, optimal_weight, optimal_bias+optimal_prompt)

    @staticmethod
    def from_linear(linear_module, rank):
        new_linear = SuperScalableLinear(linear_module.in_features, linear_module.out_features, rank)
        new_linear.weight = linear_module.weight
        new_linear.bias = linear_module.bias
        return new_linear

class ModuleInjection:

    @staticmethod
    def make_scalable(linear_module, rank=4):
        """Make a (linear) layer super scalable.
        :param linear_module: A Linear module
        :return: a suepr linear that can be trained to
        """
        new_linear = SuperScalableLinear.from_linear(linear_module, rank)
        return new_linear