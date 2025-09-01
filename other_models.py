import torch.nn as nn
import torch
import math
import torch.nn.functional as F

class ShallowConvNet(nn.Module):
    def __init__(self, C, N, nb_classes=2, NT=40, NS=40, tkerLen=12, pool_tLen=35, pool_tStep=7, batch_norm=True, dropRate=0.25, sfreq=125, tsne=False):
        super(ShallowConvNet, self).__init__()

        self.conv1 = nn.Conv2d(1, NT, (1, tkerLen), bias=False)
        self.conv2 = nn.Conv2d(NT, NS, (C, 1), bias=False)
        self.Bn1 = nn.BatchNorm2d(NS)
        self.AvgPool1 = nn.AvgPool2d((1, pool_tLen), stride=(1, pool_tStep))
        self.Drop1 = nn.Dropout(dropRate)
        fc_inSize = self.get_size(C, N)[1]
        self.classifier = nn.Linear(fc_inSize, nb_classes, bias=True)
        self.batch_norm = batch_norm
        self.tsne = tsne

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.batch_norm:
            x = self.Bn1(x)
        x = x ** 2
        x = self.AvgPool1(x)
        x = torch.log(x)
        x = self.Drop1(x)
        
        if self.tsne:
            features = x.view(x.size()[0], -1)
            return features
        
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x
    
    def get_size(self, C, N):
        data = torch.ones((1, 1, C, N))
        x = self.conv1(data)
        x = self.conv2(x)
        x = self.AvgPool1(x)
        x = x.view(x.size()[0], -1)
        return x.size()

class Oh_CNN(nn.Module):
    def __init__(self, channel, timepoint, sfreq=125, tsne=False):
        super(Oh_CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=channel, out_channels=5, kernel_size=3, stride=1)   
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)                      
        self.conv2 = nn.Conv1d(in_channels=5, out_channels=5, kernel_size=3, stride=1)   
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)       

        self.drop1 = nn.Dropout(0.5)

        self.conv3 = nn.Conv1d(in_channels=5, out_channels=5, kernel_size=3, stride=1)  
        self.avgpool1 = nn.AvgPool1d(kernel_size=2, stride=2)

        self.drop2 = nn.Dropout(0.5)

        self.conv4 = nn.Conv1d(in_channels=5, out_channels=5, kernel_size=3, stride=1)  
        self.avgpool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv1d(in_channels=5, out_channels=5, kernel_size=3, stride=1)  
        
        globalInputSize = self.get_size(channel, timepoint)[2]
        self.globalpool1 = nn.AvgPool1d(kernel_size=globalInputSize)

        self.classifier = nn.Linear(in_features=5, out_features=2)
        
        self.leakyReLU = nn.LeakyReLU()
        self.drop3 = nn.Dropout(0.5)

        self.tsne = tsne

    def forward(self, x):
        x = self.conv1(x)
        x = self.leakyReLU(x)

        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.leakyReLU(x)

        x = self.maxpool2(x)
        x = self.drop1(x)

        x = self.conv3(x)
        x = self.leakyReLU(x)

        x = self.avgpool1(x)
        x = self.drop2(x)

        x = self.conv4(x)
        x = self.leakyReLU(x)

        x = self.avgpool2(x)

        x = self.conv5(x)
        x = self.leakyReLU(x)

        x = self.globalpool1(x)
        
        if self.tsne:
            features = x.view(x.size()[0], -1)
            return features
        
        x = x.view(x.size()[0], -1)
        x = self.drop3(x)
        x = self.classifier(x)
        return x
    
    def get_size(self, channel, timepoint):
        data = torch.ones((1, channel, timepoint))
        x = self.conv1(data)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.avgpool1(x)
        x = self.conv4(x)
        x = self.avgpool2(x)
        x = self.conv5(x)

        return x.size()

class SzHNN(nn.Module):
    def __init__(self, channel, timepoint, sfreq=125, tsne=False):
        super(SzHNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=channel, out_channels=5, kernel_size=15, stride=1)

        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(in_channels=5, out_channels=10, kernel_size=10, stride=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.lstm = nn.LSTM(input_size=10, hidden_size=32, num_layers=1, batch_first=True)

        self.dense = nn.Linear(32, 64)
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(64, 2)

        self.tsne = tsne

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Reshaping for LSTM layer
        x = x.permute(0, 2, 1) 

        x, _ = self.lstm(x)
        x = x[:, -1, :]  

        x = self.dense(x)

        if self.tsne:
            features = x
            return features
        
        x = F.relu(x)
        x = self.dropout(x)

        x = self.output(x)
        x = F.softmax(x, dim=1)

        return x

# https://hackmd.io/@PKgH818nRJaPjJPqJzrGEg/SJa2dQXqel
class EEGNet(nn.Module):
    def __init__(self, channels, samples, sfreq=125, n_classes=2, F1=8, F2=16, D=2, tsne=False):
        super(EEGNet, self).__init__()

        self.tp = samples
        self.ch = channels
        self.sf = sfreq
        self.n_class = n_classes
        self.half_sf = math.floor(self.sf / 2)

        self.F1 = F1
        self.F2 = F2
        self.D = D

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.half_sf), padding='valid', bias=False), 
            nn.BatchNorm2d(self.F1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.F1, self.D * self.F1, (self.ch, 1), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.D * self.F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),  
            nn.Dropout(0.5)  
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(self.D * self.F1, self.D * self.F1, (1, math.floor(self.half_sf / 4)), padding='valid',
                      groups=self.D * self.F1, bias=False),
            nn.Conv2d(self.D * self.F1, self.F2, (1, 1), bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)), 
            nn.Dropout(0.5)
        )

        fc_inSize = self._get_size(self.ch, self.tp)[1]
        self.classifier = nn.Linear(fc_inSize, self.n_class, bias=True)

        self.tsne = tsne

    def forward(self, x):
        if len(x.shape) != 4:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.tsne:
            features = x.view(x.size()[0], -1)
            return features

        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x

    def _get_size(self, ch, tsamp):
        data = torch.ones((1, 1, ch, tsamp))
        x = self.conv1(data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size()[0], -1)
        return x.size()

class SCCNet(nn.Module):
    def __init__(self, C, N, sfreq=125.0, nb_classes=2, Nu=None, Nt=1, Nc=20, dropoutRate=0.5, tsne=False):
        super(SCCNet, self).__init__()
        Nu = C if Nu is None else Nu

        self.fs1 = int(math.floor(sfreq * 0.1))

        self.conv1 = nn.Conv2d(1, Nu, (C, Nt))
        self.Bn1 = nn.BatchNorm2d(Nu)
        self.conv2 = nn.Conv2d(Nu, Nc, (1, int(self.fs1)), padding=(0, int(self.fs1 / 2)))
        self.Bn2 = nn.BatchNorm2d(Nc)

        self.Drop1 = nn.Dropout(dropoutRate)
        self.AvgPool1 = nn.AvgPool2d((1, int(sfreq / 2)), stride=(1, int(self.fs1)))
        self.fc_inSize = self.get_size(C, N)[1]
        self.classifier = nn.Linear(self.fc_inSize, nb_classes, bias=True)

        self.tsne = tsne

    def forward(self, x):
        x = self.conv1(x)
        x = self.Bn1(x)

        x = self.conv2(x)
        x = self.Bn2(x)

        x = x ** 2
        x = self.Drop1(x)
        x = self.AvgPool1(x)

        x = torch.log(x)
        
        if self.tsne:
            features = x.view(x.size()[0], -1)
            return features
        
        x = x.view(-1, self.fc_inSize)
        x = self.classifier(x)

        return x

    def get_size(self, C, N):
        data = torch.ones((1, 1, C, N))
        x = self.conv1(data)
        x = self.Bn1(x)
        x = self.conv2(x)
        x = self.Bn2(x)
        x = self.AvgPool1(x)
        x = x.view(x.size()[0], -1)
        return x.size()

# ---- MBSzEEGNet START ----------------------------------------------------------------------------
def conv_out(sample, kernel, stride=1, padding=0):
    return math.floor((sample + 2 * padding - (kernel - 1) - 1) / stride + 1)

def pool_out(sample, kernel, stride=1, padding=0):
    return math.floor((sample + 2 * padding - kernel) / stride + 1)

def padding_same_tuple(kernel):
    kernel -= 1
    if kernel % 2 == 0:
        target = kernel / 2
    else:
        target = (kernel + 1) / 2
    target = int(target)
    if kernel % 2 == 0:
        return (target, target, 0, 0)
    else:
        return (target, target - 1, 0, 0)

class Multi(nn.Sequential):
    def __init__(self, n_classes, channels, samples, sfreq, ksize, eeg_ksize, spatial_channel, tsne, temporal_channel=20, embedded_size=8):
        super().__init__(
            dimChecker(4, 1),
            Stack(n_classes, channels, samples, sfreq, spatial_channel, temporal_channel, ksize, eeg_ksize, embedded_size),
            ClassificationHead(n_classes, samples, sfreq, spatial_channel, temporal_channel, ksize, eeg_ksize, embedded_size, tsne)
        )

    def getParamsNum(self):
        return sum(p.numel() for p in self.parameters())

class dimChecker(nn.Module):
    def __init__(self, dim, extendAt):
        super(dimChecker, self).__init__()
        self.dims = dim
        self.extendAt = extendAt
        self.inputShapeWarned = False

    def forward(self, x):
        if len(x.shape) != self.dims:
            if not self.inputShapeWarned:
                warnings.warn(f"Incorrect Input Shape {x.shape}, expected dim = {self.dims}")
                self.inputShapeWarned = True
            x = x.unsqueeze(self.extendAt)
        return x

class SCC(nn.Module):
    def __init__(self, n_classes, channels, samples, sfreq, spatial_channel, temporal_channel, ksize, embedded_size):
        super().__init__()
        kernel_conv2 = int(sfreq * ksize)
        kernel_pool = math.ceil(min(samples, sfreq * 0.5))
        self.conv1 = nn.Conv2d(1, spatial_channel, (channels, 1))

        pads = padding_same_tuple(kernel_conv2)
        self.pad2 = nn.ZeroPad2d(pads)
        self.conv2 = nn.Conv2d(spatial_channel, temporal_channel, (1, kernel_conv2))

        self.bn1 = nn.BatchNorm2d(spatial_channel)
        self.bn2 = nn.BatchNorm2d(temporal_channel)
        self.pool = nn.AvgPool2d((1, kernel_pool), (1, math.ceil(sfreq * 0.1)))
        self.Drop1 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pad2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x ** 2
        x = self.Drop1(x)
        x = self.pool(x)
        x = x + 1e-15
        x = torch.log(x)
        x = x.flatten(1)
        return x

class EEG(nn.Module):
    def __init__(self, n_classes, channels, samples, sfreq, ksize, embedded_size):
        super().__init__()
        self.tp = samples
        self.ch = channels
        self.sf = sfreq
        self.n_class = n_classes
        self.half_sf = math.floor(self.sf * ksize)

        self.F1 = 8
        self.F2 = 16
        self.D = 2
        pads = padding_same_tuple(self.half_sf)
        self.conv1 = nn.Sequential(
            nn.ZeroPad2d(pads),
            nn.Conv2d(1, self.F1, (1, self.half_sf), bias=False),  # 62,32
            nn.BatchNorm2d(self.F1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.F1, self.D * self.F1, (self.ch, 1), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.D * self.F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5)
        )

        pads = padding_same_tuple(math.ceil(self.half_sf / 4))
        self.Conv3 = nn.Sequential(
            nn.ZeroPad2d(pads),
            nn.Conv2d(self.D * self.F1, self.D * self.F1, (1, math.ceil(self.half_sf / 4)), groups=self.D * self.F1, bias=False),
            nn.Conv2d(self.D * self.F1, self.F2, (1, 1), bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.Conv3(x)
        return x

class SCCNetFC(nn.Module):
    def __init__(self, n_classes, channels, samples, sfreq, spatial_channel, temporal_channel, ksize, embedded_size):
        super().__init__()
        kernel_pool = math.ceil(min(samples, sfreq * 0.5))

        out = samples
        out = pool_out(out, kernel_pool, math.ceil(sfreq * 0.1))
        self.classifier = nn.Linear(temporal_channel * out, embedded_size)
        self.classifier_out = nn.Linear(temporal_channel * out, n_classes)

    def forward(self, x):
        x = x.flatten(1)
        out = self.classifier_out(x)
        x = self.classifier(x)
        return x, out

class EEGNetStackFC(nn.Module):
    def __init__(self, n_classes, channels, samples, sfreq, ksize, embedded_size):
        super().__init__()
        F2 = 16

        out = samples
        out = pool_out(out, 4, 4)
        out = pool_out(out, 8, 8)
        eeg_embedded = F2 * out

        self.classifier = nn.Linear(eeg_embedded, embedded_size)
        self.classifier_out = nn.Linear(eeg_embedded, n_classes)

    def forward(self, x):
        x = x.flatten(1)
        out = self.classifier_out(x)
        x = self.classifier(x)

        return x, out

class Stack(nn.Module):
    def __init__(self, n_classes, channels, samples, sfreq, spatial_channel, temporal_channel, ksize, eeg_ksize, embedded_size):
        super().__init__()
        self.stacks = nn.ModuleList([SCC(n_classes, channels, samples, sfreq, spatial_channel, temporal_channel, k, embedded_size) for k in ksize])
        self.stacks_fc = nn.ModuleList([SCCNetFC(n_classes, channels, samples, sfreq, spatial_channel, temporal_channel, k, embedded_size) for k in ksize])
        self.eeg_stacks = nn.ModuleList([EEG(n_classes, channels, samples, sfreq, k, embedded_size) for k in eeg_ksize])
        self.eeg_stacks_fc = nn.ModuleList([EEGNetStackFC(n_classes, channels, samples, sfreq, k, embedded_size) for k in eeg_ksize])
        self.klen = len(ksize) + len(eeg_ksize)

    def forward(self, x, **kwargs):
        x_output = []
        output = []
        for stack, fc in zip(self.stacks, self.stacks_fc):
            out = stack(x)
            x_out, out = fc(out)
            x_out = x_out.flatten(1)
            x_output.append(x_out)
            output.append(out)

        for stack, fc in zip(self.eeg_stacks, self.eeg_stacks_fc):
            out = stack(x)
            x_out, out = fc(out)
            x_out = x_out.flatten(1)
            x_output.append(x_out)
            output.append(out)

        x_output = torch.stack(x_output, 1)
        x_output = torch.sum(x_output, 1)

        return x_output, output

class ClassificationHead(nn.Sequential):
    def __init__(self, n_classes, samples, sfreq, spatial_channel, temporal_channel, ksize, eeg_ksize, embedded_size, tsne):
        super().__init__()
        self.classifier = nn.Linear(embedded_size, n_classes)
        self.Drop1 = nn.Dropout(0.4)

        self.tsne = tsne

    def forward(self, x):
        x_output, output = x
        x_output = self.Drop1(x_output)
        if self.tsne:
            features = x_output.view(x_output.size()[0], -1)
            return features
        x_output = self.classifier(x_output)
        return x_output

def MBSzEEGNet(channels, samples, sfreq=125.0, n_classes=2, tsne=False):
    return Multi(n_classes, channels, samples, sfreq, ksize=[0.1, 0.8], eeg_ksize=[0.1, 0.5], spatial_channel=channels, tsne=tsne)
# ---- MBSzEEGNet END ------------------------------------------------------------------------------

# ---- EEG Conformer START ----------------------------------------------------------------------------
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import torch

from PIL import Image

import torch
import torch.nn.functional as F

from torch import nn
from torch import Tensor

from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
# use conv to capture local features, instead of postion embedding.
class PatchEmbedding(nn.Module):
    def __init__(self, channels, emb_size=40):
        # self.patch_size = patch_size
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (channels, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )


    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead_C(nn.Sequential):
    def __init__(self, emb_size, n_classes, tsne):
        super().__init__()
        
        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(1440, 256), # 125 Hz, 5 sec
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )
        self.tsne = tsne

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        if self.tsne:
            return x
        out = self.fc(x)
        return out


class Conformer(nn.Sequential):
    def __init__(self, channels, samples, sfreq=125.0, n_classes=2, tsne=False, emb_size=40, depth=6, **kwargs):
        super().__init__(

            PatchEmbedding(channels, emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead_C(emb_size, n_classes, tsne)
        )
# ---- EEG Conformer END ----------------------------------------------------------------------------