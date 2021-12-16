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
from util_Data import load_rad, load_10

from util_Model2 import Resid2
import pickle

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
# DHOME="/work2/ysorimachi/mix_som/out/syn_mesh/data" #2021.10.12
DHOME="/work2/ysorimachi/mix_som/out/syn_data/data" #2021.11.22
DHOME2="/work2/ysorimachi/mix_som/out/syn_mesh/data2" #保存先

SYNFOS_INIT = "../../src/now_jst_som.txt"
CLUSTER="/home/ysorimachi/data/synfos/som/model/cluster"
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 18
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']


ERR="/home/ysorimachi/data/synfos/som/model/err"
CATE="/home/ysorimachi/data/synfos/cate"

#------------2021.10.08 --------------
def save_numpy(save_path,obj):
    save_path2 = save_path.split(".")[0]
    np.save(save_path2, obj.astype('float32'))
    return 

def load_numpy(path):
    obj = np.load(path)
    return obj

def make_cate_rule(ecode="ecmf003"):
    resid2_name="svr"
    cele = "HICA"
    cut_bins_outlier =[-1000,-500,-300,-150,0,150,300,500,1000]
    cut_bins_ci = [-1,0.2,0.4,0.6,0.9,1.5]
    
    hash_bins_outlier= {v:k for k,v in enumerate(cut_bins_outlier)}
    hash_bins_ci= {v:k for k,v in enumerate(cut_bins_ci)}
    
    def conv_cate(x):
        # idx=9999
        x = int(x.replace("(","").replace("]","").split(",")[0])
        try:
            return hash_bins_outlier[x]
        except:
            return len(hash_bins_outlier)
        # if x == cut_bins_outlier[0]:
        
    def conv_ci(x):
        x = float(x.replace("(","").replace("]","").split(",")[0])
        try:
            return hash_bins_ci[x]
        except:
            return len(hash_bins_ci)
    
    
    #------------outliers ---------------#
    if 1:
        path = f"{ERR}/max/day_max_err_{ecode}_{cele}_{resid2_name}.csv"
        opath = f"{CATE}/DAY_cate1.csv"
        df = pd.read_csv(path)
        # cut_bins =[-1000,-200,-100,-50,0,50,100,200,1000]
        df["DAY_MAX_CATE"] = pd.cut(df["MIX"],cut_bins_outlier).astype(str)
        df["DAY_MAX_CATE"] = df["DAY_MAX_CATE"].apply(lambda x: conv_cate(x))
        df.to_csv(opath,index = False)
    #------------outliers ---------------#
    if 1:
        opath = f"{CATE}/DAY_cate2_ci_mean.csv"
        df = load_rad(ecode=ecode)
        df = df.rename(columns={
            "obs" : "OBS",
            "rCR0" : "CR0",
        })

        df = cut_time(df,10,14)
        df = df.rename(columns={"day":"dd"})
        df = calc_ci_day(df,path=None)
        # df = train_flg(df)
        # print(df.head())
        # sys.exit()
        
        # ave0,ave1 = df.describe().T["min"].values[0],df.describe().T["max"].values[0]
        
        df["DAY_CI"] = pd.cut(df["mean"],cut_bins_ci).astype(str)
        df["DAY_CI"] = df["DAY_CI"].apply(lambda x: conv_ci(x))
        df.to_csv(opath,index = False)


def make_dataset(_FD=["FD2","FD2","FD2"],_CATE = ["HICA","MICA","LOCA"],name="sample"):
    _DIR = [ os.path.join(DHOME,dd) for dd in sorted(os.listdir(DHOME)) ]
    for DIR in tqdm(_DIR):
        _img = []
        dd = os.path.basename(DIR)
        for fd, cate in zip(_FD,_CATE):
            if fd=="FD2":
                fd="FD39"
            path = os.path.join(DIR,f"{fd}_{cate}_s56.npy")
            img = load_numpy(path)
            img = img.reshape(1,img.shape[0],img.shape[1])
            _img.append(img)
        
        if len(_img)==len(_CATE):
            img = np.concatenate(_img)
            save_path = f"{DHOME2}/{dd}_{name}.npy"
            save_numpy(save_path, img)
            print(datetime.now(),"[END]", dd)
        else:
            print(dd, "Not found! ")
    return


if __name__== "__main__":
    
    make_cate_rule() #cate rulr の作成
    
    #cate ---
    _FD,_CATE,name=["FD2","FD2","FD2"],["HICA","MICA","LOCA"],"cloud3"
    # _FD,_CATE,name=["FD1","FD2"],["MSPP","MSPP"],"sp_Ts2"
    make_dataset(_FD,_CATE,name)
        