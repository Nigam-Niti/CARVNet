import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial

class Bed_Network(nn.Module):
    def __init__(self,
                 sample_size,
                 sample_duration,
                 num_classes=65):

        super(Bed_Network, self).__init__()
        #Encoder
        self.group1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))
        self.group2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group5 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU())
        self.group6 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU())
        self.group7 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU())
        self.group8 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU())
        #Decoder
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(512, 512, 3, stride=(1,1,1), padding=(1,1,1))
        )
        self.temporal_reduce1 = nn.Sequential(
            nn.Conv3d(512, 512, 1, stride=(1,1,1), padding=(0,0,0))
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(512, 512, 3, stride=(1,1,1), padding=(1,1,1))
        )
        self.temporal_reduce2 = nn.Sequential(
            nn.Conv3d(128, 512, 1, stride=(4,4,4), padding=(0,0,0))
        )

        last_duration = int(math.floor(sample_duration / 8))
        last_size = int(math.ceil(sample_size / 16))
        self.fc1 = nn.Sequential(
            nn.Linear((512 * last_duration * last_size * last_size) , 4096),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc = nn.Sequential(
            nn.Linear(4096, num_classes))         

        

    def forward(self, x):
      #encoder
        out0 = self.group1(x)
        out1 = self.group2(out0)
        out2 = self.group3(out1)
        out3 = self.group4(out2)
        out4 = self.group5(out3)      
        out5 = self.group6(out4)
        out6 = self.group7(out5)
        out7 = self.group8(out6)
      #decoder
        deco1 = self.deconv1(out7)
        temp_re1 = self.temporal_reduce1(out3)
        concat1 = deco1 + temp_re1
        deco2 = self.deconv2(concat1)
        temp_re2 = self.temporal_reduce2(out1)
        concat2 = deco2 + temp_re2   
        out = concat2.view(concat2.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc(out)
        return out


def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('fc')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")
