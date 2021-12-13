# -*- coding: utf-8 -*-
# when   : 2021.07.15
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
from getErrorValues import me,rmse,mae,r2, mape # 2021.09.02
from convSokuhouData import conv_sfc #(df, ave=minutes,hour)
#amedas relate 2020.02,04 making...
from tool_AMeDaS import code2name, name2code
from tool_110570 import get_110570,open_110570
from tool_100571 import get_100571,open_100571
#(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
import subprocess
# outd ='/home/griduser/work/sori-py2/deep/out/0924_01'
# os.makedirs(outd, exist_ok=True)
# subprocess.run('rm -f *.png', cwd=outd, shell=True)
# from utils_teleme import *
sys.path.append("..")
from utils import *
from mapbox import map_lonlat3 #(df,html_path,zoom=4,size_max=None)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures


TELEME="/work/ysorimachi/hokuriku/dat2/teleme/dataset" #30min_201912.csv
SMAME_SET3="/work/ysorimachi/hokuriku/dat2/smame/set3_ts" #30min_201912.csv

def load_smame(cate,month):
  path = f"{SMAME_SET3}/{cate}_{month}.csv"
  df = pd.read_csv(path)
  df["time"] = pd.to_datetime(df["time"])
  df = df.set_index("time")
  df["p.u"] = df["sum"]*2/df["max"]
  return df

def max_df(df,cate="max"):
  """
  全カラムがテレメ地点の必要性あり
  """
  _idx = df.index
  _columns = df.columns
  df = df.replace(np.nan,-9999)
  dat = np.where(df.values == -9999,0,1)
  df2 = pd.DataFrame(dat,index=_idx,columns = _columns)
  for c in df2.columns:
    v = teleme_max(code=c,cate=cate)
    df2[c] = df2[c] * v
  return df2

def get_pu(df):
  df2 = max_df(df,cate="max")
  df3 = max_df(df,cate="panel")
  use_col = [ c for c in df.columns if "telm" in c ]
  
  df["sum"] = df[use_col].sum(axis=1)
  df["sum_max"] = df2[use_col].sum(axis=1)
  df["sum_panel"] = df3[use_col].sum(axis=1)
  df["p.u"] = df["sum"]/df["sum_max"]
  return df

def clensing(df,use_col,drop=True):
  n_before = df.shape[0]
  for c in use_col:
    df[c] = df[c].apply(lambda x: np.nan if x<0 or x>1.2 else x)
    
  if drop:
    df = df[use_col]
    df = df.dropna()
  n_after = df.shape[0]
  # print(n_before, "->", n_after)
  return df

def get_smame_rad(cate,mm,radname="obs"):
  #teleme ---
  df = load_smame(cate=cate,month=mm)
  # df = get_pu(df)
  #rad ---
  rad = load_rad(month=mm,cate=radname, lag=30)
  rad = rad["mean"]/1000 #W->Kw
  rad.name = "obs"
  df = pd.concat([df,rad],axis=1)
  return df

def get_a_b_c_d(month,cate):
  mm=month[4:6]
  if cate == "obs":
    path = "/home/ysorimachi/work/hokuriku/tbl/teleme/coef/ligression_smame_3dim_obs.csv"
  if cate == "8now0":
    """sorimachi making 2021.09.02 """
    path = "/home/ysorimachi/work/hokuriku/tbl/teleme/coef/ligression_smame_3dim_8now0.csv"
    
  df = pd.read_csv(path)
  df["mm"] = df["month"].astype(str).apply(lambda x: x[4:6])
  df = df[["mm","a","b","c","d"]].set_index("mm").T
  para = df.to_dict()
  return para[mm]["a"],para[mm]["b"],para[mm]["c"],para[mm]["d"]


def seido(radname="8now0"):
  """
  init : 2021.08.11 start(teleme) 
  update : 2021.09.02(smame 版にcopyして実装を行う予定) 
  """
  # err_model = get_err_model(err_name)
  #local function --------
  def get_pu(df):
    df2 = max_df(df,cate="max")
    df3 = max_df(df,cate="panel")
    use_col = [ c for c in df.columns if "telm" in c ]
    
    df["sum"] = df[use_col].sum(axis=1)
    df["sum_max"] = df2[use_col].sum(axis=1)
    df["sum_panel"] = df3[use_col].sum(axis=1)
    df["p.u"] = df["sum"]/df["sum_max"]
    return df
  
  def clensing(df,use_col,drop=True):
    n_before = df.shape[0]
    for c in use_col:
      df[c] = df[c].apply(lambda x: np.nan if x<0 or x>1.2 else x)
    
    use_col2 = ["sum","sum_max","sum_panel"]+use_col
    if drop:
      df = df[use_col2]
      df = df.dropna()
    # n_after = df.shape[0]
    # print(n_before, "->", n_after)
    return df

  
  def get_err(df):
    """ 2021.09.02 """
    df = df.reset_index()
    df["hh"] = df["time"].apply(lambda x: x.hour)
    
    d1 = df[(df["hh"]>=6)&(df["hh"]<=18)]
    d2 = df[(df["hh"]>=9)&(df["hh"]<=15)]
    
    # e1 = me(d1["sum"],d1["PV-max"])
    e1 = me(d1["PV-max"],d1["sum"])
    e2 = rmse(d1["sum"],d1["PV-max"])
    e3 = mape(d2["sum"],d2["PV-max"])
    return e1,e2,e3
    
    
  #local function --------
  
  _mm = loop_month(st="201904")
  # _mm19 = loop_month(st="201904")[:12]
  # print(_mm)
  # print(_mm19)
  # sys.exit()
  
  f,ax = plt.subplots(3,4,figsize=(4*5,3*4))
  ax = ax.flatten()
  
  OUTDIR="/home/ysorimachi/work/hokuriku/out/smame/pu/png"
  ERR="/home/ysorimachi/work/hokuriku/out/smame/pu/err"
  ESTIMATE="/home/ysorimachi/work/hokuriku/out/smame/pu/estimate"
  err_hash ={}
  # for i,(mm,mm19)in enumerate(zip(_mm,_mm19)):
  
  # err_hash = []
  for i ,mm in enumerate(_mm):
    # dataset 
    # mm="201910"
    
    # df = get_teleme_rad(mm,radname)
    df = get_smame_rad(cate="surplus",mm=mm,radname="8now0")
    df = df.dropna()
    # df = clensing(df,use_col = ["p.u","obs"],drop=True) #clensing
    
    # print(df.describe())
    # print(df.head())
    # sys.exit()
    # continue
    # sys.exit()
    
    # df1 = get_teleme_rad(mm19,radname)
    # df1 = clensing(df1,use_col = ["p.u","obs"],drop=True) #clensing
    
    #---------------------------------------------------------
    # ---------------------------------------
    #
    # a,b,c = get_a_b_c(mm,cate="8now0")
    a,b,c,d = get_a_b_c_d(mm,cate="8now0")
    # a1,b1,c1 = get_a_b_c(mm,cate="obs")
    
    # fitting 
    # pf = PolynomialFeatures(degree=2)
    # X2 = pf.fit_transform(df["obs"].values.reshape(-1,1))
    # lr = LinearRegression().fit(X2,df["p.u"].values)
    
    #estimate
    # df["pu-calc"] = df["obs"].apply(lambda x: a*x*x + b*x + c)
    df["pu-calc"] = df["obs"].apply(lambda x: a*x**3 + b*x**2 + c*x+d)
    # df["p.u(mm19)"] = df["obs"].apply(lambda x: a1*x*x + b1*x + c1)
    # df["p.u(mm20)"] = lr.predict(X2)
    

    df["PV-max"] = df["max"] * df["pu-calc"] 
    # df["PV(mm19)-max"] = df["sum_max"] * df["p.u(mm19)"]
    # df["PV(mm20)-max"] = df["sum_max"] * df["p.u(mm20)"]
    
    # df["PV(1905)-panel"] = df["sum_panel"] * df["p.u(1905)"] 
    # df["PV(mm19)-panel"] = df["sum_panel"] * df["p.u(mm19)"]
    # df["PV(mm20)-panel"] = df["sum_panel"] * df["p.u(mm20)"]
    
    # pv_col = [ c for c in df.columns if "-max" in c or "-panel" in c]
    pv_col = [ c for c in df.columns if "-max" in c] #2021.08.11
    
    _err=[]
    for c in pv_col:
      df[c] = df[c].apply(lambda x: np.nan if x<0 else x)
    
    df.to_csv(f"{ESTIMATE}/{mm}_{radname}.csv",index=False)
    df = df.dropna()
    
    if 0: #debug
      df["ratio"] = (df["PV-max"] - df["sum"])/(df["sum"]+0.00001)
      df=df.sort_values("ratio", ascending=False)
      print(df.head())
      sys.exit()
      
    e1,e2,e3 = get_err(df)
    err_hash[mm] = [e1,e2,e3]
    # print("2021.09.02 err start ...")
    print(datetime.now(),"[end]", mm, radname)
    
  df = pd.DataFrame(err_hash).T
  # df.index= _mm
  df.index.name="month"
  df.columns = ["me","rmse","%mae"]
  df.to_csv(f"{ERR}/only_smame_{radname}.csv")
  #---------------------------------------------------------
  return

def scatter(err_name="me",radname="obs"):
  _mm = loop_month(st="202004")
  _mm19 = loop_month(st="201904")[:12]
  
  f,ax = plt.subplots(3,4,figsize=(4*5,3*4))
  ax = ax.flatten()
  
  OUTDIR="/home/ysorimachi/work/hokuriku/out/teleme/pu/png"
  ERR="/home/ysorimachi/work/hokuriku/out/teleme/pu/err"
  ESTIMATE="/home/ysorimachi/work/hokuriku/out/teleme/pu/estimate"
  # err_hash ={}
  for i,(mm,mm19)in enumerate(zip(_mm,_mm19)):
    path = f"{ESTIMATE}/{mm}.csv"
    df = pd.read_csv(path)
    
    # pv_col = [ c for c in df.columns if "-max" in c or "-panel" in c]
    pv_col = [ c for c in df.columns if "-max" in c]
    df = df[["sum"] + pv_col]
    
    vmax = df.describe().T["max"].max()+100
    
    for c in pv_col:
      ax[i].scatter(df["sum"],df[c],label=c,s=1.5)
    
    ax[i].set_ylim(0,vmax)
    ax[i].set_xlim(0,vmax)
    ax[i].plot(np.arange(vmax),np.arange(vmax),lw=1,color="k")
    ax[i].set_ylabel("estimete[kWh]")
    ax[i].set_xlabel("OBS[kWh]")
    ax[i].legend()
    ax[i].set_title(mm)
  
  plt.subplots_adjust(wspace=0.4, hspace=0.4)
  f.savefig(f"{OUTDIR}/scatter_PV_{radname}.png",bbox_inches="tight")
  return 

def seido_plot1(err_name):
  OUTDIR="/home/ysorimachi/work/hokuriku/out/teleme/pu/png"
  ERR="/home/ysorimachi/work/hokuriku/out/teleme/pu/err"
  ESTIMATE="/home/ysorimachi/work/hokuriku/out/teleme/pu/estimate"
  p1 = f"{ERR}/{err_name}_teleme_obs.csv"
  p2 = f"{ERR}/{err_name}_teleme_8now0.csv"
  
  df = pd.read_csv(p1).drop(['PV(mm20)-max'],axis=1)
  df2 = pd.read_csv(p2).drop(['PV(1905)-max','PV(mm20)-max'],axis=1)
  df = df.merge(df2,on="month", how="inner")
  df.columns = ["month",f"{err_name}(obs-05)",f"{err_name}(obs-mm)",f"{err_name}(8Now0-mm)"]
  df["month"] = df["month"].astype(str).apply(lambda x: x[:4]+"年" + x[4:6]+"月")
  # print(df.head())
  # sys.exit()
  df.to_csv(f"{ERR}/{err_name}_result.csv", index=False,encoding="shift-jis")
  return
  


if __name__ == "__main__":
  if 1:
    # for err_name in ["rmse","me"]:
    for radname in ["8now0"]:
      seido(radname=radname)
      # sys.exit()
      # scatter(err_name="me",radname="obs")
  
  if 0:
    for err_name in ["rmse","me"]:
      seido_plot1(err_name=err_name) #2021.08.11
    