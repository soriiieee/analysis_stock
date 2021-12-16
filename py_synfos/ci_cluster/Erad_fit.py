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




def predict_calc(df,m1,m2):
    x_col = ["SYN","EC","CR0"]
    x2_col = ["70RH","85RH","70UU","70VV","LOCA","MICA"]
    # y_col = "OBS"
    X = df[x_col]
    X2 = df[x2_col]
    
    df["PRED1"] = df["MIX"] #初期値
    df["RESID1"] = 0.
    df["PRED2"] = df["MIX"] #初期値
    
    
    def clensing1(x):
        if x["PRED1"]<0:
            return 0
        elif x["PRED1"] > x["CR0"]:
            return x["CR0"]
        else:
            return x["PRED1"]
        
    def clensing2(x):
        if x["PRED1"]==0:
            return 0
        elif np.abs(x["RESID1"]) > 0.2*x["CR0"]:
            return x["PRED1"]
        elif x["PRED1"] + x["RESID1"] > x["CR0"]:
            return x["PRED1"]
        else:
            return x["PRED1"] + x["RESID1"]
    # print(X.head())
    
    # _r=[]
    # for i,r in  df.iterrows():
    #     if r["SYN"] != 9999. and r["EC"] != 9999. and r["CR0"] != 9999.:
    #         _r.append(r["MIX"])
    #     else:
    df["PRED1"] = m1.predict(X)
    df["PRED1"] = df.apply(clensing1,axis=1)
    df["RESID1"] = m2.predict(X2)
    df["PRED2"] = df.apply(clensing2,axis=1)
    return df




# def predict_rad(cele="MSPP",ecode="ecmf001",resid2_name="lgb"):
def predict_rad(ecode,n_cate,name):
    """ init 2021.10.12 
        concat 2021.10.26 stat !!
        concat 2021.11.22 stat !! 
        """
    # FIT_DIR="/home/ysorimachi/data/synfos/som/model/fitting_synfos"
    
    def cut_time(df,st,ed):
        tmp = df[(df["hh_int"]>=st)&(df["hh_int"]<=ed)]
        return tmp
    
    def details_info(df):
        n_dd = df["day"].nunique()
        tmp = cut_time(df,900,1500)
        
        ci_mean = np.mean(tmp["OBS"]/tmp["CR0"])
        ci_std = np.std(tmp["OBS"]/tmp["CR0"])
        # ratio = df[df["hh_int"]==1200].groupby("mm").count()["time"]
        return n_dd,ci_mean,ci_std
    
    def clensing(df,subset_col):
        # subset_col = x_col + [y_col]
        for c in subset_col:
            df[c] = df[c].apply(lambda x: isFloat(x))
        df = df.dropna(subset=subset_col)
        return df
    
    
    def weather_info(_dd):
        _df =[]
        for ini_j in _dd:
            df = load_weather_fcs(ecode=ecode,ini_j=ini_j,normalize=True)
            _df.append(df)
        df = pd.concat(_df,axis=0)
        return df
    
    def interpolate30min(df,_cele):
        for c in _cele:
            if c == "70RH" or c == "85RH":
                # d1[c] = d1[c].interpolate('spline', order=2)
                df[c] = df[c].interpolate('linear')
            else:
                df[c] = df[c].interpolate("linear")
        return df

    
    def load_cate_num(lbl,n_cate,name):
        
        path = f"/home/ysorimachi/data/synfos/cate/predicting/pred_{n_cate}_{name}.csv"
        df = pd.read_csv(path)
        _dd = df.loc[df["true"]==lbl,"dd"].values.tolist()
        if len(_dd)>0:
            return _dd
        else:
            return []
        
    info_hash = {}
    list_ratio = []
    coef_hash = {}
    
    _df = []
    rad = load_rad(ecode)
    
    
    path = f"/home/ysorimachi/data/synfos/cate/predicting/pred_{n_cate}_{name}.csv"
    df = pd.read_csv(path)
    N_CLS = df["true"].unique().shape[0]
    N_DAY = df.shape[0]
    # print(df["true"].value_counts())
    # sys.exit()
    
    _model=[]
    _label=[]
    _df=[]
    for lbl in range(N_CLS):
        _dd = load_cate_num(lbl,n_cate,name)
        # n,p = len(_dd),len(_dd)*100/df.shape[0]
        n = len(_dd)
        p = np.round(n*100/N_DAY, 2)  
        print(f"cate{n_cate} - {name} - N=",n,f"[{p:.3f}%]")
        
        # n,_dd = select_kmeans_day(cele=cele,kmeans_label=lbl) #64分類用
        # n0,_dd0 = select_som_day(cele=cele,lbl=lbl,istrain=True) #train
        # n1,_dd1 = select_som_day(cele=cele,lbl=lbl,istrain=False,istest=True) #test  
        # n,_dd = select_som_day(cele=cele,lbl=lbl,istrain=False,istest=False) #test
        
        
        # data srt
        if n:
            # print("CLUSTER",lbl ,f"is [ {n} ] ",cele,ecode,resid2_name)
            _dd = [ str(dd) for dd in _dd ]
            if "20180402" in _dd:
                _dd.remove("20180402")
            if "20180401" in _dd:
                _dd.remove("20180401")
            
            
            w = weather_info(_dd)
            df = rad.loc[rad["day"].isin(_dd),:]
            df = df.merge(w, on="time", how="left")
            
            # print(df.columns)
            # sys.exit()
            # _cele = ["70RH","85RH","70UU","70VV","LOCA","MICA"]
            # rad_col = ["SYN","EC"]
            rad_col = ["SYN","EC","CR0"] #正規化するとあまりにも精度が悪化するので
            if not "SYN" in df.columns:
                df = df.rename(columns = {
                    "syn": "SYN",
                    "ecm2": "EC",
                    "rCR0": "CR0",
                    "mix2": "MIX",
                    "obs": "OBS",
                }) 
            
            # wea_col = ["70RH","85RH","70UU","70VV"]
            wea_col = ['30RH', '50RH','70RH','70UU', '70VV', '85OO', '85RH', '85UU', '85VV']
            df = interpolate30min(df,wea_col) #30分内挿の実施
            
            df = cut_time(df,st=600,ed=1800)
            # df = cut_time(df,st=800,ed=1600)
            df = train_flg(df)
            # for c in rad_col:
            #     df[c] /= df["CR0"]
                
            df2 = df[df["istrain"]==1] #学習データのみ算出
            df2 = cut_time(df2,st=800,ed=1600) #更に午前中以外のデータは除外で捨ててしまう(重要度合いが極めて低い)
            #---pred1(fit)---#
            
            x_col = rad_col + wea_col
            y_col = "OBS"
            df2 = clensing(df2,subset_col = x_col+[y_col] ) #float&drop
            df = clensing(df,subset_col = x_col+[y_col] ) #　最終的な出力用のファイルもしっかりとクレンジング処理
            #---pred1(fit)---#
            X,y = df2[x_col].values,df2[y_col].values
            X1,y1 = df[x_col].values,df[y_col].values #検証含める
            # lr = LinearRegression(fit_intercept=False).fit(X,y)
            lr = LinearRegression().fit(X,y)
            # ecode,n_cate,name
            model_path = f"{MODEL1}/lr_{ecode}_cate{n_cate}_{name}_cls{lbl}.pkl"
            save_model(model_path,lr) #save
            _model.append(lr)
            _label.append(lbl)
            
            df["PRED"] = lr.predict(X1)
            df.to_csv(f"{ESTIMATE}/rad_{ecode}_cate{n_cate}_{name}_cls{lbl}.csv", index=False)
            _df.append(df) #master data setting ...
        else:
            # print("CLUSTER",lbl ,"is [ 0 ] ",cele,ecode,resid2_name)
            pass
        
        print(datetime.now(),"[END] LABEL->",lbl)
    
    df_all = pd.concat(_df,axis=0)
    df_all = df_all.sort_values("time")
    df_all = df_all.dropna(subset= x_col)
    for lbl, model in zip(_label,_model):
        df_all[f"PRED{lbl}"] = model.predict(df_all[x_col])
    df_all.to_csv(f"{ESTIMATE}/rad_{ecode}_cate{n_cate}_{name}_all.csv", index=False)
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
        if 1:
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
            
        