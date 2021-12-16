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
mms = MinMaxScaler()

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


DHOME="/work2/ysorimachi/mix_som/dat" #synfos/ data 

# OHOME="/work2/ysorimachi/mix_som/out/syn_data/data"
OHOME="/work2/ysorimachi/mix_som/out/syn_mesh/data"
SYNFOS_INIT = "../../src/now_jst_som.txt"
from util_synfos import *

def check():
    _f= sorted(glob.glob(f"{DHOME}/*_070RH*.png"))
    print(_f)
    sys.exit()

def triming(img,lonlat=[125,145,25,45],size=56):
    ny,nx = img.shape
    _lon = np.linspace(120,150,nx)
    _lat = np.linspace(20,50,ny)

    x0,x1,y0,y1 = lonlat

    iy0 = np.argmin(np.abs(_lat - y0))
    iy1 = np.argmin(np.abs(_lat - y1))
    ix0 = np.argmin(np.abs(_lon - x0))
    ix1 = np.argmin(np.abs(_lon - x1))
    
    img = img[iy0:iy1,ix0:ix1]
    img = Image.fromarray(img)
    img = img.resize((size,size))
    img = np.array(img)
    img = mms.fit_transform(img)
    return img

def load_img(cate,hh):
    chh = str(hh).zfill(2)
    path = f"{DHOME}/wrf_0{cate}_12_{chh}.png"
    img = np.array(Image.open(path))
    img = clensing(img,cate)
    return img

#------------2021.10.08 --------------
def save_numpy(save_path,obj):
    save_path2 = save_path.split(".")[0]
    np.save(save_path2, obj.astype('float32'))
    return 
def load_numpy(path):
    obj = np.load(path)
    return obj

def check_imshow():
    
    cate ="MICA"
    hh= "39"
    img = load_img(cate,hh)
    img = triming(img,lonlat=[125,145,25,45],size=56)
    img = mms.fit_transform(img)
    
    # print(img)
    # sys.exit()
    
    # print(np.nanmax(img),np.max(img))
    
    
    # print(img)
    # print(np.nanmin(img),np.min(img))
    
    # sys.exit()
    # sys.exit()
    
    f,ax = plt.subplots(figsize=(20,20))
    ax.imshow(img,vmin=0,vmax=1, cmap="seismic")
    ax.set_title(f"{cate}-{hh}")
    f.savefig("/home/ysorimachi/data/synfos/som/model/mesh/check_synfos_mesh_data.png", bbox_inches="tight")
    return

def main():
    ###
    # 2021年10月1日　synfosの対象予測時刻のファイルから参照すべき地点の情報を変動について計算していく予定
    # 2021年11月11日　要素追加
    ###
    LOG_PATH= "/home/ysorimachi/work/synfos/py/som_data/a01_02.log"
    
    # 初期時刻の読込- ---
    ini_j = synfos_inij()
    # _cate = get_cate()
    _cate = get_cate()
    # print(_cate)
    # sys.exit()
    # _hh = list(range(27,27+24)) #翌々日12Z～
    
    #翌日3(=00:00JST)
    #翌日15=3+12(=12:00JST)
    #翌々日27(=00:00JST)
    #翌々日39=27+12(=12:00JST)
    # _hh = list(range(27,27+24))
    # _hh = list([15,39]) #翌日/翌々日
    _hh = list([0,15,39]) #初期値/翌日/翌々日
    
    _ini_j = [dtinc(ini_j,4,hh) for hh in _hh]
    
    ini_j0 = str(_ini_j[0])[:8]
    
    #---Store --------------------
    ODIR=f"{OHOME}/{ini_j0}" #予測対象初期値(JST)
    os.makedirs(ODIR, exist_ok=True) #director 作成
    subprocess.run(f"rm -f {ODIR}/*.npy", shell=True)
    #-----------------------------
    _size = [56,14] #CNN/free-connect
    _ecode = list(load_10()[0])
    
    for size in _size:
        for cate in _cate:
            for k,(hh,fd) in enumerate(list(zip(_hh,_ini_j))):
                # j = k+1
                chh=str(hh).zfill(2)
                """"[summary]"
                FD1=初期値
                FD2=
                """
                img = load_img(cate,hh)
                img = triming(img,size=size) #2021.10.08
                
                save_path = f"{ODIR}/FD{chh}_{cate}_s{size}.npy"
                save_numpy(save_path,img)
                # obj = load_numpy(save_path) #debug
    
                # print(datetime.now(),"end", cate)
    mes = f"[end] {ini_j} [{cate}] {ODIR}"
    log_write(LOG_PATH,mes,init=False)
    return 

def check():
    DIR="/work2/ysorimachi/mix_som/out/syn_mesh/data/20190331"
    _path = sorted(glob.glob(f"{DIR}/*.npy"))

    f,ax = plt.subplots(4,4,figsize=(20,20))
    ax = ax.flatten()
    for i,path in enumerate(_path):
        img = load_numpy(path)
        name = path.split("/")[-1].split(".")[0]
        
        ax[i].imshow(img,vmin=0,vmax=1, cmap="seismic")
        ax[i].set_title(name)

        ax[i].tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
        plt.subplots_adjust(wspace=0.2, hspace=0.2)
    f.savefig("/home/ysorimachi/data/synfos/som/model/mesh/check_synfos_mesh_data.png", bbox_inches="tight")
    print("end")
    return
        

if __name__== "__main__":
    main() #出力していくイメージ
    # check()
    # check_imshow()