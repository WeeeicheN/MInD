from argparse import Namespace
import json
import logging
import os
import numpy as np
import random
import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple

"""
======================================================================================================
==================================== Init ====================================
======================================================================================================
"""

def boolstr(string):
    if string not in {'False','True','0','1'}:
        raise ValueError('Not a valid boolean string.')
    return (string=='True') or (string=='1')


def get_device(device):
    if torch.cuda.is_available():
        return torch.device(device)
    else:
        return torch.device("cpu")


def get_logger(logger_name, logger_dir=None, log_name=None, mute_logger=False):
    logger = logging.getLogger(logger_name)
    logger.handlers.clear() # always new

    if mute_logger:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    hterm = logging.StreamHandler() # handler for scream output
    hterm.setFormatter(formatter)
    hterm.setLevel(logging.INFO)
    logger.addHandler(hterm)

    if logger_dir is not None:
        if log_name is None:
            logger_path = os.path.join(logger_dir, f"{logger_name}.log")
        else:
            logger_path = os.path.join(logger_dir, log_name)
        hfile = logging.FileHandler(logger_path) # handler for file output
        hfile.setFormatter(formatter)
        hfile.setLevel(logging.INFO)
        logger.addHandler(hfile)
    
    return logger


def save_configs(configs, save_dir, logger, fname=None):
    if isinstance(configs, Namespace):
        configs_dict = vars(configs)
    elif isinstance(configs, object):
        configs_dict = {}
        configs_dict.update(configs.__dict__)
    else:
        assert isinstance(configs, dict)
        configs_dict = configs

    if fname is None:
        fpath = os.path.join(save_dir, 'configs.json')
    else:
        assert fname.endswith(".json")
        fpath = os.path.join(save_dir, fname)
    
    #pretrained_emb = configs_dict['pretrained_emb']
    #word2id = configs_dict['word2id']
    #del configs_dict['pretrained_emb']
    #del configs_dict['word2id']

    with open(fpath, "w") as output:
        json.dump(configs_dict, output, indent=4)
    logger.info(configs_dict)
    
    #configs_dict['pretrained_emb'] = pretrained_emb
    #configs_dict['word2id'] = word2id


def set_seed(seed):
    ## seed init.
    random.seed(seed)
    np.random.seed(seed)
    #os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

    ## torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.enabled = False # training spends longer after enabling this opts.

    ## autocheck, avoiding non-deterministic algorithms
    torch.use_deterministic_algorithms(True)


def stable(loader, seed):
    """
    Ensure ++ reproducible, ref: https://zhuanlan.zhihu.com/p/531766476?utm_id=0
    Usage:
    for epoch in range(MAX_EPOCH):  # training
        for inputs, labels in stable(DataLoader(...), seed + epoch):
            pass
    """
    set_seed(seed)
    return loader

"""
======================================================================================================
==================================== Model Outputs ====================================
======================================================================================================
"""

class MInD_Ext_Outputs(NamedTuple):
    share_v: torch.Tensor
    share_a: torch.Tensor
    share_t: torch.Tensor
    private_v: torch.Tensor
    private_a: torch.Tensor
    private_t: torch.Tensor
    noise_v: torch.Tensor
    noise_a: torch.Tensor
    noise_t: torch.Tensor
    global_mutual_share_v: torch.Tensor 
    global_mutual_share_a: torch.Tensor
    global_mutual_share_t: torch.Tensor
    global_mutual_share_v_prime: torch.Tensor
    global_mutual_share_a_prime: torch.Tensor
    global_mutual_share_t_prime: torch.Tensor
    global_mutual_private_v: torch.Tensor
    global_mutual_private_a: torch.Tensor
    global_mutual_private_t: torch.Tensor
    global_mutual_private_v_prime: torch.Tensor
    global_mutual_private_a_prime: torch.Tensor
    global_mutual_private_t_prime: torch.Tensor
    global_mutual_noise_v: torch.Tensor
    global_mutual_noise_a: torch.Tensor
    global_mutual_noise_t: torch.Tensor
    global_mutual_noise_v_prime: torch.Tensor 
    global_mutual_noise_a_prime: torch.Tensor
    global_mutual_noise_t_prime: torch.Tensor
    feat_v: torch.Tensor
    feat_a: torch.Tensor
    feat_t: torch.Tensor

"""
======================================================================================================
==================================== Functional ====================================
======================================================================================================
"""



class BTLoss(nn.Module):
    """Barlow Twins Loss"""
    def __init__(self, configs):
        super().__init__()
        self.embdim = configs.embdim
        self.bn = nn.BatchNorm1d(self.embdim, affine=False)

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x1, x2):
        batch_size = x1.size(0)
        
        c = self.bn(x1).T @ self.bn(x2)
        c = c / batch_size

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        
        loss = on_diag + (1 / self.embdim) * off_diag # From BarlowTwins-HSIC
        
        return loss



class CMDLoss(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """

    def __init__(self):
        super().__init__()

    def forward(self, x1, x2, n_moments):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1-mx1
        sx2 = x2-mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1-x2,2)
        summed = torch.sum(power)
        sqrt = summed**(0.5)
        return sqrt
        # return ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)



class DiffLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        batch_size = x1.size(0)
        x1 = x1.view(batch_size, -1)
        x2 = x2.view(batch_size, -1)

        # Zero mean
        x1_mean = torch.mean(x1, dim=0, keepdims=True)
        x2_mean = torch.mean(x2, dim=0, keepdims=True)
        x1 = x1 - x1_mean
        x2 = x2 - x2_mean

        x1_l2_norm = torch.norm(x1, p=2, dim=1, keepdim=True).detach()
        x1_l2 = x1.div(x1_l2_norm.expand_as(x1) + 1e-6)
        

        x2_l2_norm = torch.norm(x2, p=2, dim=1, keepdim=True).detach()
        x2_l2 = x2.div(x2_l2_norm.expand_as(x2) + 1e-6)

        diff_loss = torch.mean((x1_l2.t().mm(x2_l2)).pow(2))

        return diff_loss



class DJSLoss(nn.Module):
    """Jensen Shannon Divergence loss"""

    def __init__(self):
        super().__init__()

    def __call__(self, T, T_prime):
        """
        Estimator of the Jensen Shannon Divergence
        Args:
            T: Statistic network estimation from the joint distribution P(xz)
            T_prime: Statistic network estimation from the marginal distribution P(x)P(z)
        Returns:
            DJS estimation value
        """
        joint_expectation = (-F.softplus(-T)).mean()
        marginal_expectation = F.softplus(T_prime).mean()
        mutual_info = joint_expectation - marginal_expectation

        return -mutual_info



class HSICLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def distmat(self, x):
        """distance matrix"""
        r = torch.sum(x*x, 1)
        r = r.view([-1, 1])
        a = torch.mm(x, torch.transpose(x,0,1))
        D = r.expand_as(a) - 2*a +  torch.transpose(r,0,1).expand_as(a)
        D = torch.abs(D)
        
        return D

    def get_kernel(self, x, sigma=5):
        """kernel matrix, sigma taken from HSIC-bottlenect as 5"""
        n = x.size(0)
        H = torch.eye(n) - (1./n) * torch.ones([n,n])
        
        Dxx = self.distmat(x)
        variance = 2.*sigma*sigma*x.size(1)            
        Kx = torch.exp( -Dxx / variance).type(torch.FloatTensor) # Kernel Matrices        
        #print(sigma, torch.mean(Kx), torch.max(Kx), torch.min(Kx))
    
        #Kxc = torch.mm(Kx, H) # Centered Kernel Matrices

        return Kx

    def forward(self, x1, x2):
        K1 = x1 @ x1.t()
        K2 = x2 @ x2.t()
        #K1 = self.get_kernel(x1) # HSIC-BN, cause un-reproducible
        #K2 = self.get_kernel(x2)
        K12 = K1 @ K2
        n = K12.size(0)
        h = torch.trace(K12) / n**2 + torch.mean(K1) * torch.mean(K2) - 2 * torch.mean(K12) / n # From Jianlin Su
        hsic_loss = h * n**2 / (n - 1)**2
        return hsic_loss



class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p

        return output, None



class SIMSELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, - pred)
        n = torch.numel(diffs.data)
        simse = torch.sum(diffs).pow(2) / (n ** 2)

        return simse
