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
from getErrorValues import me,rmse,mape,r2,nrmse #(x,y)
#---------------------------------------------------
import subprocess
from tool_time import dtinc
# mms = MinMaxScaler()

sys.path.append("..")
from som_data.a01_99_utils import *
# from som_data.c01_som_cluster import *
from som_data.x99_pre_dataset import load_rad, load_10

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

def get_err_machine(err_name):
    if err_name=="me":
        return me
    if err_name == "rmse":
        return rmse
    if err_name=="mape":
        return mape
    if err_name == "nrmse":
        return nrmse



def load_predict_Rad(cele="MSPP",ecode="ecmf001",resid2_name="lgb",drop_element=True):
    path = f"{ESTIMATE1}/rad_{resid2_name}_{ecode}_{cele}.csv"
    df = pd.read_csv(path)
    use_col = ['time', 'OBS', 'MIX', 'SYN', 'EC', 'CR0','PRED1', 'RESID1', 'PRED2', 'CLUSTER','istrain']
    if drop_element:
        df = df[use_col]
    if df["time"].dtypes == object:
        df["time"] = pd.to_datetime(df["time"])
    df["dd"] = df["time"].apply(lambda x: x.strftime("%Y%m%d"))
    df["mm"] = df["time"].apply(lambda x: x.month)
    df["hh"] = df["time"].apply(lambda x: x.hour)
    return df

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
        # vmin,vmax = 0,250
        vmin,vmax = 0,400
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


def kaizen_rate(df,c1,c2):
    _index = df.index
    err_hash={}
    _v =[]
    
    def calc(x):
        if x[0] ==9999 or x[1]==9999:
            return np.nan
        else:
            v = - 1.0 * (x[1] - x[0])/ x[0] * 100 #%
            return v
    
    for c in c2:
        df[c] = df[[c1,c]].apply(lambda x: calc(x),axis=1)
    
    # df = df.drop([c1],axis=1)
    df[c1] = 0. #改善率=0 
    return df
        
        




# def plot_bar(err_name,cate="mm",cele="MSPP",resid2_name="lgb"):
def plot_bar(err_name,n_cate,name,err_cate):
    # _ecode,_scode,_name= load_10()
    # resid2_name="lgb"
    
    vmin,vmax,xlbl,ylbl = get_graph_setting(err_name,cate=err_cate)
    # print(vmin,vmax,xlbl,ylbl)
    # sys.exit()
    # f,ax = plt.subplots(1,10,figsize=(40,4))
    # ax = ax.flatten()
    _ecode = ["ecmf003"]
    _name = ["TOKYO"]
    for ecode,ename in zip(_ecode,_name):
        f,ax = plt.subplots(2,2,figsize=(18,10))
        for i,name_fit in enumerate(["train","test"]):
            path = f"{ERR}/{err_cate}/{ecode}_{n_cate}_{name}_{name_fit}.csv"
            # print(path)
            # sys.exit()
            
            df = load_data(path)
            df = df.replace(9999.,np.nan)
            # print(df.head())
            # sys.exit()
            
            df = df.set_index(err_cate)
            use_col = [ c for c in df.columns if c.endswith(f"_{err_name}")]
            _count = df["count"].values.tolist()
            n = np.sum(_count)
            _pc = [ np.round(v*100/n,1) for v in _count]
            
            # print(_count)
            # print(_pc)
            # sys.exit()
            df = df[use_col]
            ax[0,i] = sub_plot_bar(ax[0,i],df)
            ax[0,i].set_xlabel(xlbl)
            ax[0,i].set_ylabel(ylbl)
            ax[0,i].set_ylim(vmin,vmax)
            ax[0,i].set_title(f"{ename}(cate{n_cate})-{name_fit}")
            
            h = ax[0,i].get_ylim()[1]*0.9
            n = df.shape[0]
            for k,x0 in enumerate(_count):
                p = _pc[k]
                ax[0,i].text(k+0.25,h,f"{x0}d",fontsize=8)
                ax[0,i].text(k+0.25,h*0.9,f"{p}%",fontsize=8)
            # print(ax[0,i].get_ylim()[1]*0.9)
            # sys.exit()
            
            tmp = kaizen_rate(df,c1=f"MIX_{err_name}",c2=[f"PRED_{err_name}"])
            # print(tmp.head())
            # sys.exit()
            ax[1,i] = sub_plot_bar(ax[1,i],tmp)
            ax[1,i].set_xlabel(xlbl)
            ax[1,i].set_ylabel("Improvement Rate [%]")
            # ax[1,i].set_ylim(-50,50)
            ax[1,i].set_ylim(-200,250)
            ax[1,i].set_title(f"{ename}(cate{n_cate})-{name_fit}")
            # print(tmp.head())
            # sys.exit()
            
        # plt.subplots_adjust(wspace=0.4, hspace=0.5)
        plt.subplots_adjust(wspace=0.2, hspace=0.4)
        f.savefig(f"{ERR}/png/{err_name}_{ecode}_{err_cate}_cate{n_cate}_{name}_{name_fit}.png", bbox_inches="tight")
        print(datetime.now(), f"{ERR}/png/{ecode}_{err_name}_{err_cate}_cate{n_cate}_{name}_{name_fit}.png")
        plt.close()
        # sys.exit()
    return 


def err_png_cleaner(cate):
  subprocess.run(f"rm {ERR}/png/{cate}/*.png",shell=True)
#   subprocess.run(f"rm {ERR}/png/cls/*.png",shell=True)
#   subprocess.run(f"rm {MODEL2}/.pkl",shell=True)
  return

if __name__== "__main__":
    
    log_path = "./log_plot.log"
    log_write(log_path,"start! ",init=True)
    if 1:
        # _name = ["cloud3","sp_Ts2"]
        _name = ["cloud3"]
        _n_cate = [1,2]
        ecode = "ecmf003"
        cut_day_time = [800,1600]
        #debug ---
        # _cele = ["MSPP"]
        # _ecode = _ecode[:1]
        #debug ---
        err_name = "rmse"
        err_cate = "cls"
        
        for n_cate in _n_cate:
            for name in _name:
                plot_bar(err_name,n_cate,name,err_cate)
                # sys.exit()
                #-----------------
                print(datetime.now(),"[END]",err_name,n_cate,name)
                log_write(log_path,f"[END] {err_name},{n_cate},{name}",init=False)
                
    
