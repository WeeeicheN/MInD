import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

from backbones import *
import torch
import torch.nn as nn
from utils import MInD_Ext_Outputs



"""
======================================================================================================
==================================== Block ====================================
======================================================================================================
"""

class AutoEncoder(nn.Module):
    def __init__(self, indim, outdim, activation, dropout_rate=None):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(indim, outdim),
                activation, # nn.GELU, nn.Sigmoid(),
                nn.LayerNorm(outdim),
                nn.Dropout(dropout_rate),
                nn.Linear(outdim, outdim),
                activation)
    
    def forward(self, x):
        out = self.net(x)
        return out

class GlobalStatNet(nn.Module):
    def __init__(self, xdim, zdim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.estimator = nn.Sequential(
                        nn.Linear(xdim+zdim, 512), nn.ReLU(),
                        nn.Linear(512, 512), nn.ReLU(),
                        nn.Linear(512, 1))

    def forward(self, x, z):
        x = self.flatten(x)
        z = self.flatten(z)
        xz = torch.cat((x, z), dim=1)
        global_statistics = self.estimator(xz)
        return global_statistics

class ReverseLayerF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

"""
======================================================================================================
==================================== Model ====================================
======================================================================================================
"""

class Encoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        self.visual_backbone = Transformer(configs.indim_v, configs.embdim, nhead=configs.bbtf_vhead, nlayers=configs.bbtf_vlayers)
        self.audio_backbone = Transformer(configs.indim_a, configs.embdim, nhead=configs.bbtf_ahead, nlayers=configs.bbtf_alayers)
        self.text_backbone = Bert(configs)
           
    def forward(self, v, a, t_bert, t_bert_type, t_bert_mask):
        v = self.visual_backbone(v)
        a = self.audio_backbone(a)
        t = self.text_backbone(t_bert, t_bert_type, t_bert_mask)
        
        return v, a, t

class Extractor(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        # Backbone
        embdim = embdim_v = embdim_a = embdim_t = configs.embdim
        # MyModel
        dropout_rate = configs.enc_dropout
        # Net
        self.ae_share = AutoEncoder(embdim, embdim, configs.activation(), dropout_rate=dropout_rate)

        self.global_stat_share_v = GlobalStatNet(3*embdim, embdim_v)
        self.global_stat_share_a = GlobalStatNet(3*embdim, embdim_a)
        self.global_stat_share_t = GlobalStatNet(3*embdim, embdim_t)

        ## Comp
        self.ae_private_v = AutoEncoder(embdim_v, embdim_v, configs.activation(), dropout_rate=dropout_rate)
        self.ae_private_a = AutoEncoder(embdim_a, embdim_a, configs.activation(), dropout_rate=dropout_rate)
        self.ae_private_t = AutoEncoder(embdim_t, embdim_t, configs.activation(), dropout_rate=dropout_rate)

        self.global_stat_private_v = GlobalStatNet(embdim_v, embdim_v)
        self.global_stat_private_a = GlobalStatNet(embdim_a, embdim_a)
        self.global_stat_private_t = GlobalStatNet(embdim_t, embdim_t)

        ## Noise
        self.global_stat_noise_v = GlobalStatNet(embdim_v, embdim_v)
        self.global_stat_noise_a = GlobalStatNet(embdim_a, embdim_a)
        self.global_stat_noise_t = GlobalStatNet(embdim_t, embdim_t)

    def forward(self, v, a, t):
        # Shared.
        share_v = self.ae_share(v)
        share_a = self.ae_share(a)
        share_t = self.ae_share(t)
        
        share_v_prime = torch.cat([share_v[1:], share_v[0].unsqueeze(0)], dim=0)
        share_a_prime = torch.cat([share_a[1:], share_a[0].unsqueeze(0)], dim=0)
        share_t_prime = torch.cat([share_t[1:], share_t[0].unsqueeze(0)], dim=0)

        concat_uni = torch.cat((v, a, t), dim=1)

        global_mutual_share_v = self.global_stat_share_v(concat_uni, share_v)
        global_mutual_share_v_prime = self.global_stat_share_v(concat_uni, share_v_prime)

        global_mutual_share_a = self.global_stat_share_a(concat_uni, share_a)     
        global_mutual_share_a_prime = self.global_stat_share_a(concat_uni, share_a_prime)
             
        global_mutual_share_t = self.global_stat_share_t(concat_uni, share_t)
        global_mutual_share_t_prime = self.global_stat_share_t(concat_uni, share_t_prime)
        
        # Private.
        private_v = self.ae_private_v(v)
        private_a = self.ae_private_a(a) 
        private_t = self.ae_private_t(t) 

        private_v_prime = torch.cat([private_v[1:], private_v[0].unsqueeze(0)], dim=0)
        private_a_prime = torch.cat([private_a[1:], private_a[0].unsqueeze(0)], dim=0)
        private_t_prime = torch.cat([private_t[1:], private_t[0].unsqueeze(0)], dim=0)

        global_mutual_private_v = self.global_stat_private_v(v, private_v)
        global_mutual_private_a = self.global_stat_private_a(a, private_a)
        global_mutual_private_t = self.global_stat_private_t(t, private_t)
        global_mutual_private_v_prime = self.global_stat_private_v(v, private_v_prime)
        global_mutual_private_a_prime = self.global_stat_private_a(a, private_a_prime)
        global_mutual_private_t_prime = self.global_stat_private_t(t, private_t_prime)

        # Noise
        gn_v = torch.randn_like(v) # Gauss Noise
        gn_a = torch.randn_like(a)
        gn_t = torch.randn_like(t)

        noise_v = self.ae_private_v(gn_v) 
        noise_a = self.ae_private_a(gn_a)
        noise_t = self.ae_private_t(gn_t)


        noise_v_prime = torch.cat([noise_v[1:], noise_v[0].unsqueeze(0)], dim=0)
        noise_a_prime = torch.cat([noise_a[1:], noise_a[0].unsqueeze(0)], dim=0)
        noise_t_prime = torch.cat([noise_t[1:], noise_t[0].unsqueeze(0)], dim=0)

        global_mutual_noise_v = self.global_stat_noise_v(gn_v, noise_v)
        global_mutual_noise_a = self.global_stat_noise_a(gn_a, noise_a)
        global_mutual_noise_t = self.global_stat_noise_t(gn_t, noise_t)
        global_mutual_noise_v_prime = self.global_stat_noise_v(gn_v, noise_v_prime)
        global_mutual_noise_a_prime = self.global_stat_noise_a(gn_a, noise_a_prime)
        global_mutual_noise_t_prime = self.global_stat_noise_t(gn_t, noise_t_prime)

        # Feature
        feat_v = torch.cat((share_v, private_v), dim=1)
        feat_a = torch.cat((share_a, private_a), dim=1)
        feat_t = torch.cat((share_t, private_t), dim=1)

        return MInD_Ext_Outputs(
            share_v=share_v,
            share_a=share_a,
            share_t=share_t,
            private_v=private_v,
            private_a=private_a,
            private_t=private_t,
            noise_v=noise_v,
            noise_a=noise_a,
            noise_t=noise_t,
            global_mutual_share_v=global_mutual_share_v,
            global_mutual_share_a=global_mutual_share_a,
            global_mutual_share_t=global_mutual_share_t,
            global_mutual_share_v_prime=global_mutual_share_v_prime,
            global_mutual_share_a_prime=global_mutual_share_a_prime,
            global_mutual_share_t_prime=global_mutual_share_t_prime,
            global_mutual_private_v=global_mutual_private_v,
            global_mutual_private_a=global_mutual_private_a,
            global_mutual_private_t=global_mutual_private_t,
            global_mutual_private_v_prime=global_mutual_private_v_prime,
            global_mutual_private_a_prime=global_mutual_private_a_prime,
            global_mutual_private_t_prime=global_mutual_private_t_prime,
            global_mutual_noise_v=global_mutual_noise_v,
            global_mutual_noise_a=global_mutual_noise_a,
            global_mutual_noise_t=global_mutual_noise_t,
            global_mutual_noise_v_prime=global_mutual_noise_v_prime, 
            global_mutual_noise_a_prime=global_mutual_noise_a_prime,
            global_mutual_noise_t_prime=global_mutual_noise_t_prime,
            feat_v=feat_v,
            feat_a=feat_a,
            feat_t=feat_t,
            )

class Reconstructor(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        embdim_v = embdim_a = embdim_t = configs.embdim * 3

        self.dec_v = nn.Sequential(
                        nn.Linear(embdim_v, 256), configs.activation(),
                        nn.Linear(256, configs.embdim))
        self.dec_a = nn.Sequential(
                        nn.Linear(embdim_a, 256), configs.activation(),
                        nn.Linear(256, configs.embdim)) 
        self.dec_t = nn.Sequential(
                        nn.Linear(embdim_t, 256), configs.activation(),
                        nn.Linear(256, configs.embdim)) 
    
    def forward(self, zv, za, zt):
        v = self.dec_v(zv)
        a = self.dec_a(za)
        t = self.dec_t(zt)

        return v, a, t

class DiCyR_F2N(nn.Module):
    def __init__(self, configs):
        super().__init__()
        embdim_v = embdim_a = embdim_t = configs.embdim
        
        self.dec_feat_v = nn.Sequential(
                        nn.Linear(embdim_v*2, embdim_v), 
                        configs.activation(),
                        nn.Linear(embdim_v, embdim_v))
        self.dec_feat_a = nn.Sequential(
                        nn.Linear(embdim_a*2, embdim_a), 
                        configs.activation(),
                        nn.Linear(embdim_a, embdim_a)) 
        self.dec_feat_t = nn.Sequential(
                        nn.Linear(embdim_t*2, embdim_t), 
                        configs.activation(),
                        nn.Linear(embdim_t, embdim_t)) 
        self.dec_noise_v = nn.Sequential(
                        nn.Linear(embdim_v, embdim_v*2), 
                        configs.activation(),
                        nn.Linear(embdim_v*2, embdim_v*2))
        self.dec_noise_a = nn.Sequential(
                        nn.Linear(embdim_a, embdim_a*2), 
                        configs.activation(),
                        nn.Linear(embdim_a*2, embdim_a*2)) 
        self.dec_noise_t = nn.Sequential(
                        nn.Linear(embdim_t, embdim_t*2), 
                        configs.activation(),
                        nn.Linear(embdim_t*2, embdim_t*2))

    def forward(self, outputs, alpha):
        feat_v = outputs.feat_v
        feat_a = outputs.feat_a
        feat_t = outputs.feat_t
        noise_v = outputs.noise_v
        noise_a = outputs.noise_a
        noise_t = outputs.noise_t

        rev_feat_v = ReverseLayerF.apply(feat_v, alpha)
        rev_feat_a = ReverseLayerF.apply(feat_a, alpha)
        rev_feat_t = ReverseLayerF.apply(feat_t, alpha)
        rev_noise_v = ReverseLayerF.apply(noise_v, alpha)
        rev_noise_a = ReverseLayerF.apply(noise_a, alpha)
        rev_noise_t = ReverseLayerF.apply(noise_t, alpha)

        cyr_feat_v = self.dec_noise_v(rev_noise_v)
        cyr_feat_a = self.dec_noise_a(rev_noise_a)
        cyr_feat_t = self.dec_noise_t(rev_noise_t)
        cyr_noise_v = self.dec_feat_v(rev_feat_v)
        cyr_noise_a = self.dec_feat_a(rev_feat_a)
        cyr_noise_t = self.dec_feat_t(rev_feat_t)

        return cyr_feat_v, cyr_feat_a, cyr_feat_t, cyr_noise_v, cyr_noise_a, cyr_noise_t

class Fusion(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        embdim_v = embdim_a = embdim_t = configs.embdim * 2
        indim = embdim_v + embdim_a + embdim_t

        self.net = nn.Linear(indim, configs.fusedim)
    
    def forward(self, v, a, t):
        feature = torch.cat((v, a, t), dim=1)
        feature = self.net(feature)
        
        return feature

class Noise_Classifier(nn.Module):
    def __init__(self, configs):
        super().__init__()
        dropout = configs.clf_dropout
        ncls = configs.ncls
        indim = configs.embdim * 3

        self.head_fc1 = nn.Linear(indim, ncls)    
        self.head_fc2 = nn.Sequential(
                        nn.Linear(indim, indim//2), 
                        nn.Dropout(dropout),
                        configs.activation(),
                        nn.Linear(indim//2, ncls))
    
    def forward(self, feature, alpha):
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        pred = self.head_fc2(reverse_feature)
        
        return pred

class Classifier(nn.Module):
    def __init__(self, configs):
        super().__init__()
        dropout = configs.clf_dropout
        ncls = configs.ncls
        indim = configs.fusedim

        self.head_fc1 = nn.Linear(indim, ncls)    
        self.head_fc2 = nn.Sequential(
                        nn.Linear(indim, indim//2), 
                        nn.Dropout(dropout),
                        configs.activation(),
                        nn.Linear(indim//2, ncls))
    
    def forward(self, feature):
        pred = self.head_fc2(feature)
        
        return pred

class MyModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        
        self.enc = Encoder(configs)

        self.ext = Extractor(configs)

        self.rec = Reconstructor(configs)

        self.cyr = DiCyR_F2N(configs)

        self.fus = Fusion(configs)

        self.noise_clf = Noise_Classifier(configs)

        self.clf = Classifier(configs)
    
    def forward(self, v, a, t_bert, t_bert_type, t_bert_mask):

        self.enc_v, self.enc_a, self.enc_t = self.enc(v, a, t_bert, t_bert_type, t_bert_mask)
        
        self.outputs = self.ext(self.enc_v, self.enc_a, self.enc_t)

        disen_v = torch.cat((self.outputs.feat_v, self.outputs.noise_v), dim=1)
        disen_a = torch.cat((self.outputs.feat_a, self.outputs.noise_a), dim=1)
        disen_t = torch.cat((self.outputs.feat_t, self.outputs.noise_t), dim=1)
        
        self.rec_v, self.rec_a, self.rec_t = self.rec(disen_v, disen_a, disen_t)

        (self.cyr_feat_v, self.cyr_feat_a, self.cyr_feat_t, 
         self.cyr_noise_v, self.cyr_noise_a, self.cyr_noise_t) = self.cyr(self.outputs, self.configs.reverse_alpha)

        self.feat = self.fus(self.outputs.feat_v, self.outputs.feat_a, self.outputs.feat_t)

        concat_noise = torch.cat((self.outputs.noise_v, self.outputs.noise_a, self.outputs.noise_t), dim=1)
        
        self.noise_pred = self.noise_clf(concat_noise, self.configs.reverse_alpha)

        pred = self.clf(self.feat)

        return pred