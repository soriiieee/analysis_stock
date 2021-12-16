# -*- coding: utf-8 -*-
# when   : 2020.0x.xx
# who : [sori-machi]
# what : [ ]
#---------------------------------------------------------------------------
# basic-module
import matplotlib.pyplot as plt
import sys,os,re,glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.simplefilter('ignore')
from tqdm import tqdm
import seaborn as sns
#---------------------------------------------------
# sori -module
sys.path.append('/home/ysorimachi/tool')
# from getErrorValues import me,rmse,mae,r2 #(x,y)
# from convSokuhouData import conv_sfc #(df, ave=minutes,hour)
#amedas relate 2020.02,04 making...
# from tool_AMeDaS import code2name, name2code
# from tool_110570 import get_110570,open_110570
# from tool_100571 import get_100571,open_100571
#(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms
from torchinfo import summary #torchinfoはニューラルネットの中身を見れるのでおすすめ


seed=1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

class Scaler:
    def __init__(self,name):
        self.sc = {}
        self.name = name
    
    def fit_transform(self,data):
        for i in range(data.shape[0]):
                if self.name == "minmax":
                    self.sc[i] = MinMaxScaler()
                if self.name == "std":
                    self.sc[i] = StandardScaler()
                
                data[i,:,:] = self.sc[i].fit_transform(data[i,:,:])
        
        return data
    
    
class LeNet(nn.Module):
    def __init__(self,n_channel=1,out_size=10):
    # def __init__(self,n_channel=3):
        super(LeNet,self).__init__()
        # 入力のinput画像について56*56で入力
        #----------------
        self.n_channel = n_channel
        #N -> chan * N*56x56->N*6 chan * 28x28
        self.l1 = nn.Sequential(
            nn.Conv2d(n_channel,n_channel*6,kernel_size=5, padding=2),
            nn.BatchNorm2d(n_channel*6),
            nn.ReLU(),
            nn.MaxPool2d(2))
        #----------------
        #8chan * 28x28　->　16chan * 14x14
        self.l2 = nn.Sequential(
            nn.Conv2d(n_channel*6,n_channel*16,kernel_size=5, padding=2),
            nn.BatchNorm2d(n_channel*16),
            nn.ReLU(),
            nn.MaxPool2d(2))

        #8chan * 28x28　->　16chan * 14x14
        self.l3 = nn.Sequential(
            nn.Conv2d(n_channel*16,n_channel*48,kernel_size=5, padding=2),
            nn.BatchNorm2d(n_channel*48),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        #----------------
        #Linear 2 layer
        o1 = self._get_conv_outsize(56,5,1,2,max_pool=2)
        o2 = self._get_conv_outsize(o1,5,1,2,max_pool=2)
        o3 = self._get_conv_outsize(o2,5,1,2,max_pool=2)
        self.n_linear_in = self.n_channel * 48 * o3 * o3
        
        self.fc1 = nn.Linear(self.n_linear_in, 540)  # 3chan * 48 filter
        self.fc2 = nn.Linear(540, 84)
        self.fc3 = nn.Linear(84, out_size)
        
    def forward(self,x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        
        x = x.view(-1, self.n_linear_in) 
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def _get_conv_outsize(self,input_size, kernel_size, stride,pad, max_pool = 0):
        out_size =  ( input_size + 2* pad - kernel_size ) // stride + 1
        if max_pool:
            return out_size // max_pool
        return out_size

def tmp(batch_size =4,out_size=8):
    net = LeNet(n_channel=3,out_size=8)
    # print(net)
    summary(net,(batch_size,3,56,56)) #batch/channel/H/W
    sys.exit()
    # out_size1 = net._get_conv_outsize(56,5,1,2,max_pool=2)
    # out_size2 = net._get_conv_outsize(out_size1,5,1,2,max_pool=2)
    # out_size3 = net._get_conv_outsize(out_size2,5,1,2,max_pool=2)
    # print("1layer -> ", out_size1)
    # print("2layer -> ", out_size2)
    # print("3layer -> ", out_size3)
    # print(3 * 48 * out_size3 * out_size3 )


    
if __name__== "__main__":
    tmp(batch_size =4,out_size=8)