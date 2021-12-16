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

from a01_99_utils import *
# from c01_som_cluster import *
from util_Data import load_rad, load_10,load_predict_Rad , load_predict_Rad2
from util_PLOT import *

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
ESTIMATE1="/home/ysorimachi/data/synfos/som/model/estimate1"
ESTIMATE3="/home/ysorimachi/data/synfos/som/model/estimate3"
MODEL1="/home/ysorimachi/data/synfos/som/model/m1"
MODEL2="/home/ysorimachi/data/synfos/som/model/m2"
ERR="/home/ysorimachi/data/synfos/som/model/err"

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


def calc_mm_err(df,_col,err_path):
    
    _mm = list(range(1,12+1))
    err_hash = {}
    # _col = ['MIX','PRED1','PRED2']
    _err = ["me","rmse","mape","nrmse"]
    
    _err_col = [ f"{c}_{ename}" for ename in _err for c in _col]
    # print(_err_col)
    # sys.exit()
    for mm in _mm:
        tmp = df[df["mm"]==mm]
        
        if tmp.shape[0]>50:
            _err_v = []
            for err_col in _err_col:
                pc,err_name = err_col.split("_")
                e = get_err(tmp,"OBS",pc,err_name)
                _err_v.append(e)
        
        else:
            _err_v = [9999. for _ in range(len(_err_col))]
        
        err_hash[mm] = _err_v
    
    df = pd.DataFrame(err_hash).T
    df.index.name = "mm"
    df.columns = _err_col
    
    df.to_csv(err_path)
    return

def calc_cls_err(df,_col, err_path):
    
    # _mm = list(range(1,12+1))
    err_hash = {}
    # _col = ['MIX','PRED1','PRED2']
    _err = ["me","rmse","mape","nrmse"]
    
    _err_col = [ f"{c}_{ename}" for ename in _err for c in _col]
    # print(_err_col)
    # sys.exit()
    _count = []
    
    N_CLUSTER= df["CLUSTER"].nunique()
    # sys.exit()
    
    for lbl in range(N_CLUSTER):
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
    return


def err_make1(cele="MSPP",ecode="ecmf001",resid2_name="lgb",n_dim=3,isCNN=1):
    """[summary]
    月別やクラスタ別のデータを作成する
    init: 2021.10.12
    update 2021.11.22
    
    Args:
        cele (str, optional): [description]. Defaults to "MSPP".
        ecode (str, optional): [description]. Defaults to "ecmf001".
        resid2_name (str, optional): [description]. Defaults to "lgb".

    Returns:
        None [type]: [description]
    """
    # FIT_DIR="/home/ysorimachi/data/synfos/som/model/fitting_synfos"
    
    def cut_time(df,st,ed):
        tmp = df[(df["hh_int"]>=st)&(df["hh_int"]<=ed)]
        return tmp
    
    def clensing(df,subset_col):
        # subset_col = x_col + [y_col]
        for c in subset_col:
            df[c] = df[c].apply(lambda x: isFloat(x))
        df = df.dropna(subset=subset_col)
        return df
    
    #----------------------------
    # <--- load rad ---> 
    df = load_predict_Rad(cele,ecode,resid2_name,n_dim=3,isCNN=1)
    df2 = load_predict_Rad2(cele,ecode,resid2_name,n_dim=3,isCNN=1) #2021.11.22
    df = df.merge(df2, on="time", how="inner")
    # print(df.head())
    df = df.rename(columns= {f"PRED_{resid2_name}": "PRED3"})
    _col = ['MIX','PRED1','PRED2',"PRED3"]
    # use_col = ['time', 'OBS',"istrain","mm","dd","hh"] + ['MIX','PRED1','PRED2',f"PRED_{resid2_name}"]
    # df = df[use_col]
    # sys.exit()
    df0 = df[df["istrain"]==0]
    df1 = df[df["istrain"]==1]
    #----------------------------
    _name = ["train","test"]
    
    for i,df in enumerate([df0,df1]):
        name = _name[i]
        # train/ test
        #----------月別----
        err_path = f"{ERR}/mm/{name}_{resid2_name}_{ecode}_{cele}.csv"
        calc_mm_err(df,_col,err_path)
        # sys.exit()
        #----------クラスタ別----
        err_path = f"{ERR}/cls/{name}_{resid2_name}_{ecode}_{cele}.csv"
        calc_cls_err(df,_col,err_path)
        # sys.exit()
    # print(datetime.now(),"[END]",cele,ecode,resid2_name,n_dim,isCNN)
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
        _ecode = ["ecmf003"] #東京のみ
        # _cele = ["MICA","HICA"]
        _cele = ["HICA","MSPP"]
        _resid2 = ["lasso","ridge","svr","tree","rf","lgb"] + ["mlp3","mlp5"]
        #debug ---
        # _cele = ["MSPP"]
        # _ecode = _ecode[:1]
        err_png_cleaner("cls")
        #debug ---
        for resid2_name in _resid2:
            for ecode in _ecode:
                for cele in _cele:
                    # CALC ERROR ----------
                    err_make1(cele=cele,ecode=ecode,resid2_name=resid2_name)
                    # PLOT ERROR -------------
                    _err_name= [ "rmse" ] #"me",#rmse
                    cate = "cls" #
                    for err_name in _err_name:
                        for cate in ["cls"]:
                            plot_bar(err_name,cate,cele,resid2_name)
                    # LOGGER -------------
                    print(datetime.now(),"[END]",ecode,cele,resid2_name)
                    log_write(log_path,f"[END] {ecode} {cele} name={resid2_name}",init=False)
                    # sys.exit()
                
    
