from modules import *
import torch
import torch.nn as nn
import torchvision
from transformers import AutoModel, AutoConfig



"""
======================================================================================================
==================================== Model ====================================
======================================================================================================
"""

class ResNet18(nn.Module):
    """
    https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html
    """
    def __init__(self, configs, modality):
        super().__init__()
        self.modality = modality
        if modality == 'image':
            self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif modality =='audio':
            self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif modality == 'video':
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif modality == 'audio-video':
            self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            raise NotImplementedError('Incorrect modality, should be audio or visual but got {}'.format(modality))
        
        resnet18 = torchvision.models.resnet18()
        self.inconv = nn.Sequential(self.conv1, resnet18.bn1, resnet18.relu, resnet18.maxpool)
        self.layer1 = resnet18.layer1
        self.layer2 = resnet18.layer2
        self.layer3 = resnet18.layer3
        self.layer4 = resnet18.layer4
        self.avgpool = resnet18.avgpool
    
    def forward(self, x):
        if self.modality == 'video':
            (B, C, T, H, W) = x.size()
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            x = x.view(B * T, C, H, W)
        elif self.modality == 'audio-video':
            (B, C, T, H, W) = x.size()
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            x = x.view(B * T, C, H, W)
        
        x = self.inconv(x)  # B,1,28,28 -> B,64,7,7
        x = self.layer1(x)  # B,64,7,7
        x = self.layer2(x)  # B,128,4,4
        x = self.layer3(x)  # B,256,2,2
        x = self.layer4(x)  # B,512,1,1
        x = self.avgpool(x).flatten(1) # B,512,1,1 -> B,512,1,1 -> B,512

        #if self.modality == 'audio':
        #    x = x.unsqueeze(2)
        #elif self.modality =='image':
        #    x = x.unsqueeze(2)
        
        return x



class RNN(nn.Module):
    def __init__(self, configs, modality, nblocks=2, outfeat='o'):
        super().__init__()
        self.configs = configs
        self.outfeat = outfeat

        if modality == 'video':
            indim = configs.indim_v
            embdim = configs.embdim
            nlayers = configs.vrnn_layers
            dropout = configs.vrnn_dropout
        elif modality == 'audio':
            indim = configs.indim_a
            embdim = configs.embdim
            nlayers = configs.arnn_layers
            dropout = configs.arnn_dropout
        elif modality == 'text':
            indim = configs.indim_t
            embdim = configs.embdim
            nlayers = configs.trnn_layers
            dropout = configs.trnn_dropout
        
        if configs.dataset == 'AVMNIST':
            if modality == 'image':
                inchannels = configs.inchannels_v
                outchannels = configs.outchannels_v
            elif modality == 'audio':
                inchannels = configs.inchannels_a
                outchannels = configs.outchannels_a

            self.conv = nn.Conv1d(inchannels, outchannels, kernel_size=1)
        
        rnn = nn.LSTM if configs.rnntype == "LSTM" else nn.GRU
        batch_first = True if outfeat == 'o' else False
        
        self.rnn_list = [rnn(input_size=indim, hidden_size=embdim,
                            num_layers=nlayers, dropout=dropout, bidirectional=True, batch_first=batch_first)]
        self.ln_list = [nn.LayerNorm((embdim*2, ))]
        for i in range(1, nblocks):
            self.rnn_list.append(rnn(input_size=embdim*2, hidden_size=embdim,
                                num_layers=nlayers, dropout=dropout, bidirectional=True, batch_first=batch_first))
            self.ln_list.append(nn.LayerNorm((embdim*2, )))
        
        self.net = nn.ModuleList(self.rnn_list)
        self.ln = nn.ModuleList(self.ln_list)
        
        if outfeat == 'o':
            self.proj = nn.Linear(2*embdim*nblocks, embdim)
        elif outfeat == 'h':
            self.proj = nn.Linear(embdim*nblocks, embdim)
        
    def forward(self, x):
        configs = self.configs
        outfeat = self.outfeat
        
        if configs.dataset == 'AVMNIST':
            x = x.view(x.shape[0], x.shape[1], -1)
            x = self.conv(x)

        if outfeat == 'o':
            if configs.rnntype == 'LSTM':
                for i, (block, ln) in enumerate(zip(self.net, self.ln)):
                    if i == 0:
                        o, (h, _) = block(x)
                        o = ln(o)
                        x = o                        
                    else:
                        o, (h, _) = block(o)
                        o = ln(o)
                        x = torch.cat((x, o), dim=2)
                x = self.proj(x)

        elif outfeat == 'h':
            x = x.permute([1,0,2])
            if configs.rnntype == 'LSTM':
                for i, (block, ln) in enumerate(zip(self.net, self.ln)):
                    if i == 0:
                        o, (h, _) = block(x)
                        o = ln(o)
                        x = h                            # [2,B,embdim]
                    else:
                        o, (h, _) = block(o)
                        o = ln(o)
                        x = torch.cat((x, h), dim=2)
                x = self.proj(x).permute([1,0,2])
                #x = x.permute(1, 0, 2).contiguous().view(batch_size, -1)   # [B,2,embdim] -> [B,-1]
        
        return x



class Transformer_Simple(nn.Module):
    """
    Extends nn.Transformer.
    """
    def __init__(self, indim, embdim, nhead, nlayers, dropout=0.1, pe=False, use_seq=True):
        super().__init__()
        self.embdim = embdim
        self.pe = pe
        self.use_seq = use_seq

        self.conv = nn.Conv1d(indim, embdim, kernel_size=1, padding=0, bias=False)
        layer = nn.TransformerEncoderLayer(embdim, nhead=nhead, dropout=dropout) # Default batch_first = False!
        self.transformer = nn.TransformerEncoder(layer, num_layers=nlayers)

    def forward(self, x):
        """
        Apply transorformer to input tensor.
        """
        if type(x) is list:
            x = x[0]
        
        x = self.conv(x.permute([0,2,1]))
        x = x.permute([2,0,1])

        if self.pe == True:
            pass
        else:
            if self.use_seq:
                x = self.transformer(x)
            else:
                #x = self.transformer(x)[0] # AGM
                x = self.transformer(x)[-1] # MultiBench
        
        return x



class Bert(nn.Module):
    def __init__(self, configs):
        super().__init__()
        bertpath = configs.bertpath
        bertconfig = AutoConfig.from_pretrained(bertpath, output_hidden_states=True)
        self.bertmodel = AutoModel.from_pretrained(bertpath, config=bertconfig)
        self.linear = nn.Sequential(
                    nn.Linear(768, configs.embdim), 
                    nn.ReLU(),
                    nn.LayerNorm(configs.embdim))

        self.conv = nn.Conv1d(768, configs.embdim, kernel_size=3, stride=1, padding=1, bias=False)
    
    def forward(self, t_bert, t_bert_type, t_bert_mask):
        bert_output = self.bertmodel(input_ids=t_bert, token_type_ids=t_bert_type, attention_mask=t_bert_mask)
        bert_output = bert_output[0]
        
        # Pooling
        ## Masked Mean, same as MISA
        masked_output = torch.mul(t_bert_mask.unsqueeze(2), bert_output)
        mask_len = torch.sum(t_bert_mask, dim=1, keepdim=True)  
        output = torch.sum(masked_output, dim=1, keepdim=False) / mask_len
        output = self.linear(output)
        
        return output



class Transformer(nn.Module):
    def __init__(self, indim, embdim, nhead=5, nlayers=3, 
                attn_dropout=0.5, embed_dropout=0.2, relu_dropout=0.3, res_dropout=0.3,
                attn_mask=False):
        super().__init__()

        self.conv = nn.Conv1d(indim, embdim, kernel_size=1, padding=1, bias=False)
        self.transformer = TransformerEncoder(embed_dim=embdim, num_heads=nhead, layers=nlayers,
                                            attn_dropout=attn_dropout, embed_dropout=embed_dropout, 
                                            relu_dropout=relu_dropout, res_dropout=res_dropout,
                                            attn_mask=attn_mask)

    def forward(self, x):
        x = x.transpose(1, 2) # batch, indim, seq
        x = self.conv(x) # batch, embdim, seq+2
        x = x.permute(2, 0, 1) # seq+2, batch, embdim
        x = self.transformer(x) # seq+2, batch, embdim
        x = x[-1] # batch, embdim

        return x