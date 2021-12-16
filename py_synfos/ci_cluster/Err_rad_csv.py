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
rcParams['font.size'] = 18
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']


FIT_DIR="/home/ysorimachi/data/synfos/som/model/fitting_synfos"
ESTIMATE1="/home/ysorimachi/data/synfos/som/model/estimate2"
# MODEL1="/home/ysorimachi/data/synfos/som/model/m1"
# MODEL2="/home/ysorimachi/data/synfos/som/model/m2"
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
    
        

def get_err(df,tc,pc,err_name):
    """ 2021.09.02 """
    """ 2021.10.04 update """
    df = df.reset_index()
    # df["hh"] = df["time"].apply(lambda x: x.hour)

    if err_name == "mape":
        df = df[(df["hh"]>=9)&(df["hh"]<=15)]
    else:
        df = df[(df["hh"]>=6)&(df["hh"]<=18)]

    if df.shape[0] !=0:
        err = get_err_machine(err_name)(df[tc],df[pc])
    else:
        err = 9999.
    return err


def get_N_CLUS(n_cate):
    if n_cate == 1:
        N_CLS=8
    elif n_cate==2:
        N_CLS=5
    return N_CLS


def load_predict_Rad(ecode,n_cate,name,drop_element=False):
    
    N_CLS = get_N_CLUS(n_cate)
    
    _df = [] 
    for lbl in range(N_CLS):
        path = f"{ESTIMATE1}/rad_{ecode}_cate{n_cate}_{name}_cls{lbl}.csv"
        df = pd.read_csv(path)
        df["CLUSTER"] = lbl
        _df.append(df)
    
    df = pd.concat(_df,axis=0)
    
    use_col = ['time', 'OBS', 'MIX', 'SYN', 'EC', 'CR0','PRED','CLUSTER','istrain']
    if drop_element:
        df = df[use_col]
    if df["time"].dtypes == object:
        df["time"] = pd.to_datetime(df["time"])
    df["dd"] = df["time"].apply(lambda x: x.strftime("%Y%m%d"))
    df["mm"] = df["time"].apply(lambda x: x.month)
    df["hh"] = df["time"].apply(lambda x: x.hour)
    return df

def load_predict_Rad_ENS(ecode,n_cate,name,drop_element=False):    
    # N_CLS = get_N_CLUS(n_cate)
    
    # _df = [] 
    # for lbl in range(N_CLS):
    #     path = f"{ESTIMATE1}/rad_{ecode}_cate{n_cate}_{name}_cls{lbl}.csv"
    #     df = pd.read_csv(path)
    #     df["CLUSTER"] = lbl
    #     _df.append(df)
    
    path = f"/home/ysorimachi/data/synfos/som/model/estimate2/rad_{ecode}_cate{n_cate}_{name}_ENS.csv"
    df = pd.read_csv(path)
    # df = pd.concat(_df,axis=0)
    
    # use_col = ['time', 'OBS', 'MIX', 'SYN', 'EC', 'CR0','PRED','CLUSTER','istrain']
    # if drop_element:
    #     df = df[use_col]
    if df["time"].dtypes == object:
        df["time"] = pd.to_datetime(df["time"])
    df["dd"] = df["time"].apply(lambda x: x.strftime("%Y%m%d"))
    df["mm"] = df["time"].apply(lambda x: x.month)
    df["hh"] = df["time"].apply(lambda x: x.hour)
    return df


def calc_mm_err(df, err_path):
    
    _mm = list(range(1,12+1))
    err_hash = {}
    _col = ['MIX','PRED']
    # _col = ['MIX','PRED_ENS'] #2021.11.22
    _err = ["me","rmse","mape","nrmse"]
    
    _err_col = [ f"{c}_{ename}" for ename in _err for c in _col]
    _count=[]
    # print(_mm)
    # sys.exit()
    
    for mm in _mm:
        tmp = df[df["mm"]==1]
        
        if tmp.shape[0]>0:
            _err_v = []
            for err_col in _err_col:
                pc,err_name = err_col.split("_")
                e = get_err(tmp,"OBS",pc,err_name)
                _err_v.append(e)
            # print(tmp.shape)
            # print(tmp.head())
            # sys.exit()
            _count.append(tmp["dd"].nunique())
        
        else:
            _count.append(0)
            _err_v = [9999. for _ in range(len(_err_col))]
        
        err_hash[mm] = _err_v
    
    df = pd.DataFrame(err_hash).T
    df.index.name = "mm"
    df.columns = _err_col
    df["count"] = _count
    df.to_csv(err_path)
    print(err_path)
    return

def calc_cls_err(df,cate, err_path):
    
    N_CLS = get_N_CLUS(cate)
    # _mm = list(range(1,12+1))
    err_hash = {}
    # _col = ['MIX','PRED1','PRED2']
    _col = ['MIX','PRED'] #2021.10.26
    # _col = ['MIX','PRED_ENS'] #2021.11.22
    _err = ["me","rmse","mape","nrmse"]
    
    _err_col = [ f"{c}_{ename}" for ename in _err for c in _col]
    # print(_err_col)
    # sys.exit()
    _count = []
    for lbl in range(N_CLS):
        tmp = df[df["CLUSTER"]==lbl]
        # print(tmp.shape)

        if tmp.shape[0]>0:
            _err_v = []
            _count.append(tmp["dd"].nunique())
            for err_col in _err_col:
                pc,err_name = err_col.split("_")
                e = get_err(tmp,"OBS",pc,err_name)
                _err_v.append(e)
        
        else:
            _err_v = [9999. for _ in range(len(_err_col))]
            _count.append(0)
        err_hash[lbl] = _err_v
    
    df = pd.DataFrame(err_hash).T
    df.index.name = "cls"
    df.columns = _err_col
    df["count"] = _count
    df.to_csv(err_path)
    print(err_path)
    return




def err_make1(ecode,n_cate,name,cut_day_time=True):
    """
    < concat >  
        2021.10.12 stat !!
        2021.10.26 stat !! update 2class
        2021.11.22 stat !! update 2class
    """
    # FIT_DIR="/home/ysorimachi/data/synfos/som/model/fitting_synfos"
    
    def cut_time(df,st,ed):
        try:
            tmp = df[(df["hh_int"]>=st)&(df["hh_int"]<=ed)]
        except:
            st = st//100
            ed = ed//100
            tmp = df[(df["hh"]>=st)&(df["hh"]<=ed)]
        return tmp
    
    def clensing(df,subset_col):
        # subset_col = x_col + [y_col]
        for c in subset_col:
            df[c] = df[c].apply(lambda x: isFloat(x))
        df = df.dropna(subset=subset_col)
        return df
    
    #----------------------------
    # <--- load rad ---> 
    # df = load_predict_Rad(ecode,n_cate,name)
    df = load_predict_Rad_ENS(ecode,n_cate,name) #2021.11.22
    df = train_flg(df)
    df = df.rename(columns = {
        "LBL": "CLUSTER",
        "PRED_ENS": "PRED",
    })
    # print(df.head())
    # print(df["mm"].unique(), df.shape)
    # sys.exit()
    # ----------------------------
    
    if cut_day_time:
        st,ed = cut_day_time
        df = cut_time(df, st,ed)
        df["dd"] = df["time"].apply(lambda x: x.strftime("%Y%m%d"))
    
    df0 = df[df["istrain"]==1]
    df1 = df[df["istrain"]==0]
    
    # print(df0[["OBS","MIX","PRED"]].head(20))
    # print(df0.shape,df1.shape)
    # print(df1.columns)
    # sys.exit()
    #----------------------------
    _name_fit = ["train","test"]
    for i,df in enumerate([df0,df1]):
        name_fit = _name_fit[i]
        # train/ test
        #----------月別----
        err_path = f"{ERR}/mm/{ecode}_{n_cate}_{name}_{name_fit}.csv"
        calc_mm_err(df,err_path)
        # sys.exit()
        #----------クラスタ別----
        err_path = f"{ERR}/cls/{ecode}_{n_cate}_{name}_{name_fit}.csv"
        calc_cls_err(df,n_cate,err_path)
        # sys.exit()
    return 

# _img,_time = mk_DataLoader("sp",_
# def rad_cleaner():
#   subprocess.run(f"rm {ESTIMATE1}/*.csv",shell=True)
#   subprocess.run(f"rm ./*.out",shell=True)
# #   subprocess.run(f"rm {MODEL2}/.pkl",shell=True)
#   return
    


if __name__== "__main__":
        
    if 1:
        # ecode="ecmf001"
        #----------------------------------------
        log_path = "./log_err.log"
        log_write(log_path,"start! ",init=True)
        # _cele = ["MSPP","LOCA","MICA","HICA"]
        # _name = ["cloud3","sp_Ts2"]
        _name = ["cloud3"]
        _n_cate = [1,2]
        ecode = "ecmf003"
        cut_day_time = [800,1600]
        #debug ---
        # _cele = ["MSPP"]
        # _ecode = _ecode[:1]
        #debug ---
        for n_cate in _n_cate:
            for name in _name:
                err_make1(ecode,n_cate,name,cut_day_time=cut_day_time)
                
                #-----------------
                print(datetime.now(),"[END]",ecode,n_cate,name)
                log_write(log_path,f"[END] {ecode} {n_cate} name={name}",init=False)
                # sys.exit()
                
    
