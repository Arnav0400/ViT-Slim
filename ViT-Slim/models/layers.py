import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., num_patches=197):
        super().__init__()
        self.num_heads = num_heads
        self.num_patches = num_patches
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SparseAttention(Attention):
    def __init__(self, attn_module, head_search=False, uniform_search=False):
        super().__init__(attn_module.qkv.in_features, attn_module.num_heads, True, attn_module.scale, attn_module.attn_drop.p, attn_module.proj_drop.p)
        self.is_searched = False
        self.num_gates = attn_module.qkv.in_features // self.num_heads
        if head_search:
            self.zeta = nn.Parameter(torch.ones(1, 1, self.num_heads, 1, 1))
        elif uniform_search:
            self.zeta = nn.Parameter(torch.ones(1, 1, 1, 1, self.num_gates))
        else:
            self.zeta = nn.Parameter(torch.ones(1, 1, self.num_heads, 1, self.num_gates))
        self.searched_zeta = torch.ones_like(self.zeta)
        self.patch_zeta = nn.Parameter(torch.ones(1, self.num_patches, 1)*3)
        self.searched_patch_zeta = torch.ones_like(self.patch_zeta)
        self.patch_activation = nn.Tanh()
    
    def forward(self, x):
        z_patch = self.searched_patch_zeta if self.is_searched else self.patch_activation(self.patch_zeta)
        x *= z_patch
        B, N, C = x.shape
        z = self.searched_zeta if self.is_searched else self.zeta
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # 3, B, H, N, d(C/H)
        qkv *= z
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) # B, H, N, d

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def get_zeta(self):
        return self.zeta, self.patch_activation(self.patch_zeta)
    
    def compress(self, threshold_attn):
        self.is_searched = True
        self.searched_zeta = (self.zeta>=threshold_attn).float()
        self.zeta.requires_grad = False
        
    def compress_patch(self, threshold_patch=None, zetas=None):
        self.is_searched = True
        zetas = torch.from_numpy(zetas).reshape_as(self.patch_zeta)
        self.searched_patch_zeta = (zetas).float().to(self.zeta.device)
        self.patch_zeta.requires_grad = False

    def decompress(self):
        self.is_searched = False
        self.zeta.requires_grad = True
        self.patch_zeta.requires_grad = True

    def get_params_count(self):
        dim = self.qkv.in_features
        active = self.searched_zeta.sum().data
        if self.zeta.shape[-1] == 1:
            active*=self.num_gates
        elif self.zeta.shape[2] == 1:
            active*=self.num_heads
        total_params = dim*dim*3 + dim*3
        total_params += dim*dim + dim
        active_params = dim*active*3 + active*3
        active_params += active*dim +dim
        return total_params, active_params
    
    def get_flops(self, num_patches, active_patches):
        H = self.num_heads
        N = num_patches
        n = active_patches
        d = self.num_gates
        sd = self.searched_zeta.sum().data
        if self.zeta.shape[-1] == 1: # Head Elimination
            sd*=self.num_gates
        elif self.zeta.shape[2] == 1: # Uniform Search
            sd*=self.num_heads
        total_flops = N * (H*d * (3*H*d)) + 3*N*H*d #linear: qkv
        total_flops += H*N*d*N + H*N*N #q@k
        total_flops += 5*H*N*N #softmax
        total_flops += H*N*N*d #attn@v
        total_flops += N * (H*d * (H*d)) + N*H*d #linear: proj
        
        active_flops = n * (H*d * (3*sd)) + 3*n*sd #linear: qkv
        active_flops += n*n*sd + H*n*n #q@k
        active_flops += 5*H*n*n #softmax
        active_flops += n*n*sd #attn@v
        active_flops += n * (sd * (H*d)) + n*H*d #linear: proj
        return total_flops, active_flops

    @staticmethod
    def from_attn(attn_module, head_search=False, uniform_search=False):
        attn_module = SparseAttention(attn_module, head_search, uniform_search)
        return attn_module

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SparseMlp(Mlp):
    def __init__(self, mlp_module):
        super().__init__(mlp_module.fc1.in_features, mlp_module.fc1.out_features, mlp_module.fc2.out_features, act_layer=nn.GELU, drop=mlp_module.drop.p)
        self.is_searched = False
        self.num_gates = mlp_module.fc1.out_features
        self.zeta = nn.Parameter(torch.ones(1, 1, self.num_gates))
        self.searched_zeta = torch.ones_like(self.zeta)  
    
    def forward(self, x, patch_zeta=None):
        if patch_zeta is not None:
            x*=patch_zeta
        z = self.searched_zeta if self.is_searched else self.get_zeta()
        x = self.fc1(x)
        x = self.act(x)
        x *= z # both fc1 and fc2 dimensions eliminated here
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    def get_zeta(self):
        return self.zeta
    
    def compress(self, threshold):
        self.is_searched = True
        self.searched_zeta = (self.get_zeta()>=threshold).float()
        self.zeta.requires_grad = False

    def decompress(self):
        self.is_searched = False
        self.zeta.requires_grad = True

    def get_params_count(self):
        dim1 = self.fc1.in_features
        dim2 = self.fc1.out_features
        active_dim2 = self.searched_zeta.sum().data
        total_params = 2*(dim1*dim2) + dim1 + dim2
        active_params = 2*(dim1*active_dim2) + dim1 + active_dim2
        return total_params, active_params
    
    def get_flops(self, num_patches, active_patches):
        total_params, active_params = self.get_params_count()
        return total_params*num_patches, active_params*active_patches

    @staticmethod
    def from_mlp(mlp_module):
        mlp_module = SparseMlp(mlp_module)
        return mlp_module

class ModuleInjection:
    method = 'full'
    searchable_modules = []

    @staticmethod
    def make_searchable_attn(attn_module, head_search=False, uniform_search=False):
        if ModuleInjection.method == 'full':
            return attn_module
        attn_module = SparseAttention.from_attn(attn_module, head_search, uniform_search)
        ModuleInjection.searchable_modules.append(attn_module)
        return attn_module

    @staticmethod
    def make_searchable_mlp(mlp_module):
        if ModuleInjection.method == 'full':
            return mlp_module
        mlp_module = SparseMlp.from_mlp(mlp_module)
        ModuleInjection.searchable_modules.append(mlp_module)
        return mlp_module