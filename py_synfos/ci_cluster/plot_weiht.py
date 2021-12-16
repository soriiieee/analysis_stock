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
from utils_log import log_write #(path,mes,init=False)
#---------------------------------------------------
import subprocess
from tool_time import dtinc
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
mms = MinMaxScaler()

sys.path.append("/home/ysorimachi/work/synfos/py/som_data")
from a01_99_utils import *
from c01_som_cluster import *
from x99_pre_dataset import load_rad, load_10

from util_Model2 import Resid2
import pickle

from layers import LeNet

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    from torch.utils.data import DataLoader, TensorDataset, Dataset
    from torchvision import transforms
    from torchinfo import summary #torchinfoはニューラルネットの中身を見れるのでおすすめ
except:
    print("Not Found Troch Modules ...")


# DHOME="/work2/ysorimachi/mix_som/dat" #synfos/ data 

# OHOME="/work2/ysorimachi/mix_som/out/syn_data/data"
# OHOME="/work2/ysorimachi/mix_som/out/syn_mesh/data"
DHOME="/work2/ysorimachi/mix_som/out/syn_mesh/data"
DHOME2="/work2/ysorimachi/mix_som/out/syn_mesh/data2" #保存先

SYNFOS_INIT = "../../src/now_jst_som.txt"
CLUSTER="/home/ysorimachi/data/synfos/som/model/cluster"
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 18
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']


ERR="/home/ysorimachi/data/synfos/som/model/err"
CATE="/home/ysorimachi/data/synfos/cate"

FITTING="/home/ysorimachi/data/synfos/cate/fitting"
PREDICT="/home/ysorimachi/data/synfos/cate/predicting"
PNG="/home/ysorimachi/data/synfos/cate/png"

def plot_weight(name="cloud3",dd="20201009"):
    
    f,ax = plt.subplots(1,2,figsize=(22,8))
    _title = ["OutLier predict(8)" ,"CI predict(5)"]
    
    for i,n_cate in enumerate([1,2]):
        
        csv_path = f"{PREDICT}/pred_{n_cate}_{name}.csv"
        df = pd.read_csv(csv_path)
        df["dd"] = df["dd"].astype(str)
        _w = df[df["dd"]==dd]["pred"].values[0].replace("[","").replace("]","").split(",")
        _w2 = list(map(float, _w))
        
        # print(_w2)
        ax[i].bar(np.arange(len(_w2)), np.array(_w2))
        ax[i].set_ylim(0,1)
        ax[i].set_xlabel("CLUSTER N")
        ax[i].set_title(_title[i])
    
    f.savefig(f"{PNG}/weight_cnn_{name}_{dd}.png", bbox_inches="tight")
    print(f"{PNG}/weight_cnn_{name}_{dd}.png")
    plt.close()
    return        
    
    


if __name__== "__main__":
    # make_cate_rule() #cate rulr の作成
    
    #cate ---
    _FD,_CATE,name=["FD2","FD2","FD2"],["HICA","MICA","LOCA"],"cloud3"
    # _FD,_CATE,name=["FD1","FD2"],["MSPP","MSPP"],"sp_Ts2"
    n_cate=1
    
    _dd = [
        "20190329",
        "20200413",
        "20201015",
        "20201017",
        "20200909"
    ]
    
    for dd in _dd:
        plot_weight(name,dd)
    
    #---cehck---
    # check()
    
        