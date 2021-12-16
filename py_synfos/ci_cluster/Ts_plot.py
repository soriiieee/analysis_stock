# -*- coding: utf-8 -*-
# when   : 2020.0x.xx
# who : [sori-machi]
# what : [ ]
import sys,os,re,glob
Fname = sys.argv[0].split(".")[0]    
log_path = f"./log_{Fname}.log"
#---------------------------------------------------------------------------
# basic-module
import matplotlib.pyplot as plt
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
from getErrorValues import me,rmse,mape,r2,nrmse #(x,y)
#---------------------------------------------------
import subprocess
from tool_time import dtinc
# mms = MinMaxScaler()

sys.path.append("..")
from som_data.a01_99_utils import *
# from som_data.c01_som_cluster import *
from som_data.util_Data import load_rad, load_10

# from util_Model2 import Resid2
import pickle


DHOME="/work2/ysorimachi/mix_som/out/syn_mesh/data"

SYNFOS_INIT = "../../src/now_jst_som.txt"
CLUSTER="/home/ysorimachi/data/synfos/som/model/cluster"
SOM_DIR="/home/ysorimachi/data/synfos/som/model/labels_synfos/train"


from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']


FIT_DIR="/home/ysorimachi/data/synfos/som/model/fitting_synfos"
ESTIMATE1="/home/ysorimachi/data/synfos/som/model/estimate1"
MODEL1="/home/ysorimachi/data/synfos/som/model/m1"
MODEL2="/home/ysorimachi/data/synfos/som/model/m2"
ERR="/home/ysorimachi/data/synfos/som/model/err_LeNet"


from Err_rad_csv import load_predict_Rad

def get_graph_setting(err_name,cate="mm"):
    if cate=="mm":
        xlbl = "MONTH"
    elif cate=="cls":
        xlbl = "CLUSTER(N)"
    elif cate == "time":
        xlbl = r"OBS[W/$m^2$]"
        
    if err_name=="me":
        # vmin,vmax = -100,100
        vmin,vmax = -200,200
        ylbl = r"ME[W/$m^2$]"
    if err_name == "rmse":
        vmin,vmax = 0,250
        ylbl = r"RMSE[W/$m^2$]"
    if err_name=="mape":
        vmin,vmax = 0,10
        ylbl= "%MAP[-]"
    if err_name == "nrmse":
        vmin,vmax = 0,10
        ylbl = "NRMSE[-]"
    list_detail=[ vmin,vmax,xlbl,ylbl]
    return list_detail

def load_data(path):
    df = pd.read_csv(path)
    # print(df.head())
    return df


def sub_plot_bar(ax,df):
    w = np.round(1.0/(len(df.columns)+1),2)
    _index = df.index
    """
    ax : matplotlib
    df : index/cは表示するカラムのみ
    """ 
    for i,c in enumerate(df.columns):
      ax.bar(w/2 + np.arange(len(df))+w*i,df[c],width=w,label=c,align="edge")
    
    ax.set_xlim(0,len(df))
    ax.set_xticks(np.arange(len(df))+1/2)
    for x0 in np.arange(len(df)):
      ax.axvline(x=x0,color="gray", alpha=0.5, lw=1)
      
    ax.set_xticks(np.arange(len(df))+1/2)
    ax.set_xticklabels(_index,rotation=0)
    return ax


def kaizen_rate(df,c1,c0):
    
    df[c1] = np.abs(df[c1])
    df[c0] = np.abs(df[c0])
    
    def calc(x):
        if x[0] ==9999 or x[1]==9999:
            return np.nan
        else:
            v = - 1.0 * (x[0] - x[1])/ x[1] * 100 #%
            return v

    df["rate"] = df[[c1,c0]].apply(lambda x: calc(x),axis=1)
    return df["rate"].values.tolist()

def clensing_seido(df,err_name):
    if err_name == "diff":
        for c in ["MIX","PRED"]:
            df[c] -= df["OBS"]
            df[f"ABS_{c}"] = np.abs(df[c])
    else:
        pass
    return df

def select_DD(err_name,err_cate,n_cate,name):
    # vmin,vmax,xlbl,ylbl = get_graph_setting(err_name,cate=err_cate)
    if err_name == "rmse":
        names = ["rmse0","rmse1","mape0","mape1"]
    else:
        names = ["diff0","diff1","std0","std1"]
        
    rad = load_predict_Rad("ecmf003",n_cate,name).sort_values("time")
    rad = clensing_seido(rad, err_name) #error別に事前に計算を回しておく
    # print("before->", df.shape[0])
    rad = rad[rad["istrain"]==0]
    # print(vmin,vmax,xlbl,ylbl)
    df = cut_time(rad, st=8,ed=16)
    
    _dd = df["dd"].unique()
    _cls = []
    e_hash = {}
    for dd in _dd:
        tmp = df[df["dd"]==dd]
        _cls.append(tmp["CLUSTER"].values[0])
        if err_name == "rmse":
            e0  = rmse(tmp["OBS"], tmp["MIX"])
            e1  = rmse(tmp["OBS"], tmp["PRED"])
            tmp = cut_time(tmp, st=9,ed=15)
            e00 = mape(tmp["OBS"], tmp["MIX"])
            e11 = mape(tmp["OBS"], tmp["PRED"])
            
            _e = [e0,e1,e00,e11]
        else:
            _e = []
            _s = []
            for c in ["MIX","PRED"]:
                tmp = tmp.sort_values(f"ABS_{c}", ascending=False)
                e = tmp[c].values[0]
                _e.append(e)
                
                st = np.std(tmp[c])
                _s.append(st)
            
            _e += _s
            
        e_hash[dd] = _e
    
    df = pd.DataFrame(e_hash).T
    df.index.name = "dd"
    df.columns = names
    df["CLUSTER"] = _cls
    return df

def day_rad_ts(df,dd):
    df = df[df["dd"]==dd]
    return df

def plot_dd(err_name,err_cate,n_cate,name):
    ODIR="/home/ysorimachi/data/synfos/tmp/ci/ts/00"
    
    rad = load_predict_Rad("ecmf003",n_cate,name)
    # print(rad.head())
    # print(rad.columns)
    # sys.exit()
    df = select_DD(err_name,err_cate,n_cate,name)
    df.columns = ["e0","e1","s0","s1", "CLUSTER"]
    df["rate"] = kaizen_rate(df,"e1","e0")
    df = df.sort_values("rate",ascending=False)
    
    # print(df.head())
    # sys.exit()
    
    def plt_ts_dd(ax,df,out_png):
        _main = ['OBS','MIX','PRED']
        _sub = ["SYN","EC","CR0"]
        df = df.set_index("time")
        for c in _main:
            ax.plot(np.arange(0,len(df)),df[c],label = c, lw=5)
        for c in _sub:
            ax.plot(np.arange(0,len(df)),df[c],label = c, lw=1)
        
        ax.legend(loc= "upper left")
        ax.set_ylim(-100,1200)
        ax.set_xlabel("hh")
        ax.set_xlim(0,len(df))
        _index = [ time.strftime("%H:%M") for time in  df.index ]
        
        #区間色塗り
        ax.axvspan(0, 4, color="gray", alpha=0.5)
        ax.axvspan(20, len(df), color="gray", alpha=0.5)
        
        
        ax.set_xticks(np.arange(0,len(df)))
        ax.set_xlim(0,len(df))
        ax.set_xticklabels(_index, fontsize=12,rotation=45)
        
        return ax
    
    for j,(dd,r) in enumerate(list(df.iterrows())[:3]):
    # for j,(dd,r) in enumerate(list(df.iterrows())[-3:]):
        # print(j)
        # sys.exit()
        # j= len(df) - j
        ctop_num = str(j+1).zfill(5)
        clst = int(r["CLUSTER"])
        rate = np.round(r["rate"],2)
        e0,e1 = np.round(r["e0"],1),np.round(r["e1"],1)
        
        t = day_rad_ts(rad, dd)
        out_png = f"{ODIR}/{ctop_num}_{err_name}_cate{n_cate}_{name}.png"
        f,ax = plt.subplots(figsize=(15,6))
        ax = plt_ts_dd(ax,t,out_png)
        ax.set_title(f"DAY[{dd}(test)] - ecode003(TOKYO)- {ctop_num}_{err_name}_cate{n_cate}_{name}")
        
        ax.text(6,1100, f"CLUSTER{clst}() {err_name} ({e0})->({e1}) {rate}%")
        plt.savefig(out_png, bbox_inches="tight")

        print("end", j)
    return

def png_cleaner():
  subprocess.run(f"rm /home/ysorimachi/data/synfos/tmp/ci/ts/00/*.png",shell=True)
#   subprocess.run(f"rm {ERR}/png/cls/*.png",shell=True)
#   subprocess.run(f"rm {MODEL2}/.pkl",shell=True)
  return

if __name__== "__main__":
    
    log_write(log_path,"start! ",init=True)
    png_cleaner()
    if 1:
        _name = ["cloud3"]
        # _name = ["sp_Ts2"]
        # _n_cate = [1,2]
        _n_cate = [1,2]
        ecode = "ecmf003"
        cut_day_time = [800,1600]
        #debug ---
        # _cele = ["MSPP"]
        # _ecode = _ecode[:1]
        #debug ---
        err_name = "rmse" #"diff","rmse"
        err_cate = "cls"
        
        for n_cate in _n_cate:
            for name in _name:
                plot_dd(err_name,err_cate,n_cate,name)
                # sys.exit()
                #-----------------
                log_write(log_path,f"[END] {err_name},{n_cate},{name}",init=False)
        sys.exit()
                
    
