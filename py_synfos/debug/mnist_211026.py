# -*- coding: utf-8 -*-
# when   : 2020.03.23 
# who : [sori-machi]
# what : [ ]
"""
https://qiita.com/fukuit/items/215ef75113d97560e599
"""
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
from convSokuhouData import conv_sfc #(df, ave=minutes,hour)
# #amedas relate 2020.02,04 making...
from tool_AMeDaS import code2name, name2code, name2scode
# from tool_110570 import get_110570,open_110570
# from tool_100571 import get_100571,open_100571
#(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
import subprocess
# outd ='/home/griduser/work/sori-py2/deep/out/0924_01'
# os.makedirs(outd, exist_ok=True)
# subprocess.run('rm -f *.png', cwd=outd, shell=True)

#scipy module import 
# import scipy.stats as ss 
# from util_Model2 import Resid2
import pickle
# from layers import LeNet
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    
    import torchvision
    from torch.utils.data import DataLoader, TensorDataset, Dataset
    from torchvision import transforms
    from torchinfo import summary #torchinfoはニューラルネットの中身を見れるのでおすすめ
except:
    print("Not Found Troch Modules ...")
    
DATA="/home/ysorimachi/work/sori_py2/deepl/dat/data/"

def dataset():
  transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))
  ])
  
  train_set = torchvision.datasets.MNIST(root=DATA, train=True,download=True,transform=transform)
  test_set = torchvision.datasets.MNIST(root=DATA, train=False,download=True,transform=transform)
  
  print(len(train_set))
  print(train_set[1])
  print(train_set)
  print(dir(train_set[1]))
  sys.exit()
  
  
  train_loader = DataLoader(train_set, batch_size=100, shuffle=True,num_workers = 2) 
  test_loader = DataLoader(test_set, batch_size=100, shuffle=False,num_workers = 2) 

  return train_loader, test_loader


def main():
  train_loader , test_loader = dataset()
  
  for img,lbl in train_loader:
    print(img.size())
    print(lbl.size())
    sys.exit()


if __name__ == "__main__":
  main()
  # test2()
  