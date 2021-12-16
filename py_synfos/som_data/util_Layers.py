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
    
    
class CNN_FOR_SOM(nn.Module):
    def __init__(self, out_size=128):
        super(CNN_FOR_SOM,self).__init__()
        # 入力のinput画像について56*56で入力
        #----------------
        #1chan * 56x56->8chan * 28x28
        self.l1 = nn.Sequential(
            nn.Conv2d(1,8,kernel_size=5, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2))
        #----------------
        #8chan * 28x28　->　16chan * 14x14
        self.l2 = nn.Sequential(
            nn.Conv2d(8,16,kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        #----------------
        #16chan * 14x14 -> 32 * 7x7
        self.l3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        #----------------
        #Linear 2 layer
        self.l4 = nn.Sequential(
            nn.Linear(32*7*7,512),
            nn.ReLU(),
            nn.Linear(512,128),
            # nn.ReLU(),
            # nn.Linear(128,4)
            )
        
    def forward(self,x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        
        x = x.view(x.size(0),-1)
        x = self.l4(x)
        return x


        

    
if __name__== "__main__":
    main()