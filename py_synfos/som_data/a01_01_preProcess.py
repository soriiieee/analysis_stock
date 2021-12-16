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

from a01_99_utils import *

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
from util_synfos import *

DHOME="/work2/ysorimachi/mix_som/dat" #synfos/ data 

OHOME="/work2/ysorimachi/mix_som/out/syn_data/data"
SYNFOS_INIT = "../../src/now_jst_som.txt"

def check():
    _f= sorted(glob.glob(f"{DHOME}/*_070RH*.png"))
    print(_f)
    sys.exit()
    

def load_img(cate,hh):
    path = f"{DHOME}/wrf_0{cate}_12_{hh}.png"
    img = np.array(Image.open(path))
    img = clensing(img,cate)
    return img

def get_value(cate,img):
    ny,nx = img.shape
    _lat = np.linspace(20,50,ny)
    _lon = np.linspace(120,150,nx)
    
    df = load_10() #地点面
    dl = def_dl(cate) #ずらし
    _m,_s =[],[]
    for i, r in df.iterrows():
        iy = np.argmin(np.abs(_lat - r[2]))
        ix = np.argmin(np.abs(_lon - r[1]))
        m = np.mean(img[iy-dl:iy+dl,ix-dl:ix+dl])
        s = np.std(img[iy-dl:iy+dl,ix-dl:ix+dl])
        
        # m = np.round(m,2) #2桁 2021.09.30
        # s = np.round(s,2) #2桁
        m = np.round(m,4) #4桁 2021.09.30
        s = np.round(s,4) #4桁
        _m.append(m)
        _s.append(s)
    return _m,_s

def main():
    ###
    # 2021年10月1日　synfosの対象予測時刻のファイルから参照すべき地点の情報を変動について計算していく予定
    ###
    LOG_PATH= "/home/ysorimachi/work/synfos/py/som_data/a01_01.log"
    
    # 初期時刻の読込- ---
    ini_j = synfos_inij()
    _cate = get_cate()
    # print(ini_j, _cate)
    # sys.exit()
    _hh = list(range(27,27+24)) #　翌々日0時からデータを取得するようなprogram
    _ini_j = [dtinc(ini_j,4,hh) for hh in _hh]
    # print(_ini_j)
    # sys.exit()
    
    ini_j0 = str(_ini_j[0])[:8]
    ODIR=f"{OHOME}/{ini_j0}"
    os.makedirs(ODIR, exist_ok=True) #director 作成
    
    _ecode = list(load_10()[0])
    for cate in _cate:
        m_hash={}
        s_hash = {}
        for hh,ini_j in zip(_hh,_ini_j):
        
            img = load_img(cate,hh)
            _mean,_std = get_value(cate,img)
            
            m_hash[ini_j] = _mean
            s_hash[ini_j] = _std
        
        df_m = pd.DataFrame(m_hash).T
        df_s = pd.DataFrame(s_hash).T
        
        df_m.columns = _ecode
        df_s.columns = _ecode

        df_m.to_csv(f"{ODIR}/{cate}_mean.csv")
        df_s.to_csv(f"{ODIR}/{cate}_std.csv")

        # print(datetime.now(),"end", cate)
        mes = f"[end] {ini_j} [{cate}] {ODIR}"
        log_write(LOG_PATH,mes,init=False)
    return 

def file_check():
    _dd = sorted(os.listdir(OHOME))
    for dd in _dd:
        _f = os.listdir(os.path.join(OHOME,dd))
        
        if len(_f) <16:
            print(dd)
        
        # print(_f)
    sys.exit()

if __name__== "__main__":
    main() #出力していくイメージ
    # file_check()