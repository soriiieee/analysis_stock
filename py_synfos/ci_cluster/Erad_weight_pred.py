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
# mms = MinMaxScaler()
sys.path.append("..")
from som_data.a01_99_utils import *
# from som_data.c01_som_cluster import *
from som_data.util_Data import load_rad, load_10,load_predict_Rad , load_predict_Rad2,load_weather_fcs
# from util_Data import load_rad, load_10,load_predict_Rad , load_predict_Rad2
from sklearn.linear_model import LinearRegression
# from util_Model2 import Resid2
import pickle

# DHOME="/work2/ysorimachi/mix_som/dat" #synfos/ data 

# OHOME="/work2/ysorimachi/mix_som/out/syn_data/data"
# OHOME="/work2/ysorimachi/mix_som/out/syn_mesh/data"
DHOME="/work2/ysorimachi/mix_som/out/syn_mesh/data"

SYNFOS_INIT = "../../src/now_jst_som.txt"
CLUSTER="/home/ysorimachi/data/synfos/som/model/cluster"
SOM_DIR="/home/ysorimachi/data/synfos/som/model/labels_synfos/train"


from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 18
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']


FIT_DIR="/home/ysorimachi/data/synfos/som/model/fitting_synfos"
# ESTIMATE="/home/ysorimachi/data/synfos/som/model/estimate1"
ESTIMATE="/home/ysorimachi/data/synfos/som/model/estimate2"
MODEL1="/home/ysorimachi/data/synfos/cate/reg"
# MODEL2="/home/ysorimachi/data/synfos/som/model/m2"
PREDICT_WEIGHT="/home/ysorimachi/data/synfos/cate/predicting"


def load_ensemble_rad(n_cate):
    # _csv = sorted(list(glob.glob(f"{ESTIMATE}/*_cate{n_cate}_*_all.csv")))
    path = sorted(list(glob.glob(f"{ESTIMATE}/*_cate{n_cate}_*_all.csv")))[0]
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    if "PRED" in df.columns:
        df = df.drop(["PRED"],axis=1)
    
    use_col = ["time","OBS","MIX"] + sorted([c for c in df.columns if "PRED" in c])
    df = df[use_col]
    df["dd"] = df["time"].apply(lambda x: x.strftime("%Y%m%d"))
    return df

def load_weight(n_cate):
    path = f"{PREDICT_WEIGHT}/pred_{n_cate}_cloud3.csv"
    df = pd.read_csv(path)

    df = df[["dd","pred","true"]].set_index("dd")
    hash_weight = {}
    hash_cls = {}
    for dd,r in df.iterrows():
        dd = str(dd)
        _cw = r["pred"].replace("[","").replace("]","").split(",")
        _w = list(map(float, _cw))
        
        hash_weight[dd] = _w
        hash_cls[dd] = r["true"]
        
    # print(hash_weight)
    return hash_weight,hash_cls


# def predict_rad(cele="MSPP",ecode="ecmf001",resid2_name="lgb"):
def predict_rad(ecode,n_cate,name):
    """ init 2021.10.12 
        concat 2021.10.26 stat !!
        concat 2021.11.22 stat !! 
        """
    # FIT_DIR="/home/ysorimachi/data/synfos/som/model/fitting_synfos"
    WEIGHT_HASH ,LBL_HASH = load_weight(n_cate)

    rad = load_ensemble_rad(n_cate=n_cate)
    _mem_col = sorted([c for c in rad.columns if "PRED" in c])
    
    def cut_time(df,st,ed):
        tmp = df[(df["hh_int"]>=st)&(df["hh_int"]<=ed)]
        return tmp
    
    def clensing(df,subset_col):
        # subset_col = x_col + [y_col]
        for c in subset_col:
            df[c] = df[c].apply(lambda x: isFloat(x))
        df = df.dropna(subset=subset_col)
        return df
    
    def pred_ens(x):
        dd = x["dd"]
        w =  WEIGHT_HASH[dd]

        r = 0
        for i,c in enumerate(_mem_col):
            r += x[c]*w[i]
            # print(i,r)
        
        if r<0:
            r=0
        elif r>1367:
            r = 1367
        else:
            pass
        return r
        
    rad["PRED_ENS"] = rad.apply(pred_ens,axis=1)
    rad["LBL"] = rad["dd"].apply(lambda x: LBL_HASH[x]) 
    
    #  ecode,n_cate,name
    csv_path = f"{ESTIMATE}/rad_{ecode}_cate{n_cate}_{name}_ENS.csv"
    rad.to_csv(csv_path, index=False)
    return

    
# _img,_time = mk_DataLoader("sp",_
def rad_cleaner():
  subprocess.run(f"rm {ESTIMATE}/*.csv",shell=True)
  subprocess.run(f"rm ./*.out",shell=True)
#   subprocess.run(f"rm {MODEL2}/.pkl",shell=True)
  return
    


if __name__== "__main__":
        
    if 1:
        # ecode="ecmf001"
        if 0:
            rad_cleaner()
        #----------------------------------------
        log_path = "./log_predict.log"
        log_write(log_path,"start! ",init=True)
        # _cele = ["MSPP","LOCA","MICA","HICA"]
        # _ecode,_scode,_name= load_10()
        # _ecode = ["ecmf003"] #東京のみ
        _name = ["cloud3"]
        _n_cate = [1,2]
        ecode = "ecmf003"
        #debug ---
        # _cele = ["MSPP"]
        # _ecode = _ecode[:1]
        #debug ---
        for n_cate in _n_cate:
            for name in _name:
                predict_rad(ecode,n_cate,name)
                
                #------log ---------message ---
                print(datetime.now(),"[END]", ecode, n_cate, name)
                log_write(log_path,f"[END] {ecode}, {n_cate}, {name}",init=False)
                # sys.exit()
            
        