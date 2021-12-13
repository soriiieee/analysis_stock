# -*- coding: utf-8 -*-
# when   : 2021.07.15
# when   : 2021.09.02 "8Now0 にて回帰したものを用いる"
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
from getErrorValues import me,rmse,mae,r2 #(x,y)
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

sys.path.append("..")
try:
  from utils_teleme import *
except:
  from teleme.utils_teleme import *
from utils import *

try:
  from utils_smame import *
except:
  from smame.utils_smame import *

from mapbox import map_lonlat3 #(df,html_path,zoom=4,size_max=None)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures


from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 14
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']


TELEME="/work/ysorimachi/hokuriku/dat2/teleme/dataset" #30min_201912.csv

def load_teleme(month,min=30):
  path = f"{TELEME}/{min}min_{month}.csv"
  df = pd.read_csv(path)
  df["time"] = pd.to_datetime(df["time"])
  df = df.set_index("time")
  # df.to_csv("/home/ysorimachi/work/hokuriku/dat/rad/r2/rad/tmp_check2.csv")
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

OUTDIR="/home/ysorimachi/work/hokuriku/out/teleme/pu/png"
def estimate(radname="obs",CSV=False,ONLY_DATA=False,mm=False):
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
      
    if drop:
      df = df[use_col]
      df = df.dropna()
    n_after = df.shape[0]
    # print(n_before, "->", n_after)
    return df
  
  def get_teleme_rad(mm,radname="obs"):
    # mm="202103"
    #teleme ---
    df = load_teleme(mm)
    df = get_pu(df)
    #rad ---
    # radname="obs"
    rad = load_rad(month=mm,cate=radname, lag=30)
    # print(df.shape, rad.shape)
    # rad.to_csv("/home/ysorimachi/work/hokuriku/dat/rad/r2/rad/tmp_check.csv")
    # sys.exit()
    # print(rad.head())
    # sys.exit()
    rad = rad["mean"]/1000 #W->Kw
    rad.name = "obs"
    df = pd.concat([df,rad],axis=1)

    return df
  #local function --------
  #- sub 2021.09.02
  if ONLY_DATA and mm:
    return get_teleme_rad(mm,radname)
  
  #commands 
  _mm = loop_month(st="202004")
  _mm19 = loop_month(st="201904")[:12]
  # print(_mm)
  # print(_mm19)
  # sys.exit()
  
  f,ax = plt.subplots(3,4,figsize=(4*5,3*4))
  ax = ax.flatten()
  
  for i,(mm,mm19)in enumerate(zip(_mm,_mm19)):
    # dataset 
    df = get_teleme_rad(mm,radname)
    df = clensing(df,use_col = ["p.u","obs"],drop=True) #clensing

    df1 = get_teleme_rad(mm19,radname)
    df1 = clensing(df1,use_col = ["p.u","obs"]) #clensing
    # print(mm,df.shape)
    
    # print(df.describe())
    # print(mm,df.isnull().sum())
    # print(mm,df.shape)
    # continue
    # sys.exit()
    # # continue
    
    #---------------------------------------------------------
    # ---------------------------------------
    #
    a0,b0,c0 = get_a_b_c("201905",cate="obs")
    # a0,b0,c0 = get_a_b_c("201905",cate="8now0")
    a1,b1,c1 = get_a_b_c(mm,cate="obs")
    a2,b2,c2 = get_a_b_c(mm,cate="8now0")
    
    # print(a1,b1,c1)
    # print(a2,b2,c2)
    # sys.exit()
    
    # fitting 
    pf = PolynomialFeatures(degree=2)
    lr = LinearRegression()
    X2 = pf.fit_transform(df["obs"].values.reshape(-1,1))
    lr = LinearRegression().fit(X2,df["p.u"].values)
    # sys.exit()
    
    #estimate
    # df["p.u(201905)"] = df["obs"].apply(lambda x: a0*x*x + b0*x + c0)
    # df[f"p.u({mm})"] = df["obs"].apply(lambda x: a1*x*x + b1*x + c1)
    # df["p.u(2020ver)"] = lr.predict(X2)
    
    ax[i].scatter(df1["obs"],df1["p.u"],color="green", s=2, label="p.u.2019")
    ax[i].scatter(df["obs"],df["p.u"],color="k", s=2, label="p.u.2020")
    # ax[i].scatter(df["obs"],df["p.u(201905)"],color="b", s=1, label="p.u(201905-obs)")
    # ax[i].scatter(df["obs"],df[f"p.u({mm})"],color="r", s=1, label="p.u(perMonth-obs)")
    # ax[i].scatter(df["obs"],df["p.u(2020ver)"],color="gray", s=1, label="p.u(2020ver)")
    
    # line
    X_line = np.linspace(0,1,1000)
    pu_pred0 = a0*X_line**2 + b0*X_line + c0
    pu_pred1 = a1*X_line**2 + b1*X_line + c1
    pu_pred2 = lr.predict(pf.fit_transform(X_line.reshape(-1,1)))
    if 0:
      ax[i].plot(X_line,pu_pred0,lw=1,color="k", label="201905")
      ax[i].plot(X_line,pu_pred1,lw=1,color="b",label="per-Month")
      ax[i].plot(X_line,pu_pred2,lw=1,color="r", label="2020-ver")

    ax[i].legend(loc="upper left",fontsize=10)
    
    cate_name="双方向端末"
    ax[i].set_xlabel(f"日射量({radname})[kW/m2]")
    ax[i].set_ylabel(f"p.u({cate_name})")
    ax[i].set_title(f"{mm}(rad-{radname})")

    print(datetime.now(),"[end]", mm, radname)
    # sys.exit()
    #---------------------------------------------------------
  
  plt.subplots_adjust(wspace=0.4, hspace=0.4)
  f.savefig(f"{OUTDIR}/scatter_pu_{radname}.png",bbox_inches="tight")
  print(OUTDIR)
  return



def plot_pu_line():
  """ 
  2021.09.02 
  2021.09.08 with- smame も実装済 
  
  """
  f,ax = plt.subplots(3,4,figsize=(4*5,3*4))
  ax = ax.flatten()
  _mm19 = loop_month(st="201904")[:12]
  rcParams['font.size'] = 14
  
  for i,mm in enumerate(_mm19):
    
    # line
    X_line = np.linspace(0,1,1000)
    _label = ["19年05月(unyo)","月別(unyo)","月別(8now0)","月別(合算)(8now0)"]
    
    for j ,lbl in enumerate(_label):
      
      if j==0:
        month, cate= "201905","obs"
        color,alpha,ls= "gray",0.8,"--"
      if j==1:
        month, cate= mm,"obs"
        color,alpha,ls= "green",0.8,"-"
      if j==2:
        month, cate= mm,"8now0"
        color,alpha,ls= "blue",1,"-"
      if j==3:
        month, cate= mm,"8now0_with_smame"
        color,alpha,ls= "red",1,"-"
        
      a,b,c = get_a_b_c(month,cate=cate)
      
      pu_pred0 = a*X_line**2 + b*X_line + c
      ax[i].plot(X_line,pu_pred0,lw=1,color=color, label=lbl, alpha=alpha, linestyle = ls)
      # pu_pred1 = a1*X_line**2 + b1*X_line + c1
      # pu_pred2 = lr.predict(pf.fit_transform(X_line.reshape(-1,1)))
    # if 1:
    #   ax[i].plot(X_line,pu_pred0,lw=1,color="k", label="201905")
    #   ax[i].plot(X_line,pu_pred1,lw=1,color="b",label="per-Month")
    #   ax[i].plot(X_line,pu_pred2,lw=1,color="r", label="2020-ver")
    
    ax[i].legend(loc="lower right",fontsize=8)
    ax[i].set_title(f"{mm}(p.u reg)")
    
    ax[i].set_ylim(0,1)
    ax[i].set_xlim(0,1)
    ax[i].set_ylabel("p.u.")
    ax[i].set_xlabel("平均日射量[kW/m2]")
    print(datetime.now(),"[end]", mm)
    # sys.exit()
    #---------------------------------------------------------
  plt.subplots_adjust(wspace=0.4, hspace=0.4)
  f.savefig(f"{OUTDIR}/line_pu_2rad_all2.png",bbox_inches="tight")
  print(OUTDIR)
  return 

def reg_hosei_ratio(cate="Z1"):
  _mm=loop_month()[:12]
  
  #-----------
  def get_kenshin(cate,mm):
    path = f"/home/ysorimachi/work/hokuriku/dat/kenshin/re_Get/kenshin_{mm[2:6]}.csv"
    df = pd.read_csv(path)
    df["cate"] = df["番号"].apply(lambda x: str(x)[:1])
    df = df[df["cate"]=="Z"]
    return df
  
  def read_result(mm,p=None):
    if p==None:
      cp="0.0"
    else:
      cp=str(p)
    path =f"/home/ysorimachi/work/hokuriku/dat/kenshin/teleme_19_result/0_ronbun_use{cp}.csv"
    df = pd.read_csv(path)
    df["month"] = df["month"].astype(str)
    
    tmp = df[df["month"]==mm]
    pv_ken, pv_est = tmp["ken_pv"].values[0],tmp["est_pv"].values[0]
    return pv_ken, pv_est
  
  def calc_PV(cate,mm):
    ken = get_kenshin(cate,mm) #月ごとの検針記録
    pv_max = np.sum(ken["受給最大電力"])
    pv_ken = np.sum(ken["買取電力量"]) #OBS-PV
    
    a,b,c = get_mm_abc_Z(mm[4:6], radname="8now0")
    a,b,c = get_mm_abc_Z("05", radname="obs")
    rad = load_rad(month=mm,cate="obs", lag=30)
    rad = rad["mean"]/1000 #W->Kw
    rad.name = "obs"
    rad = rad.reset_index()
    rad["PV"] = (a * rad["obs"]**2 + b*rad["obs"] + c) * pv_max /2
    
    rad["PV"] = rad["PV"].apply(lambda x: 0 if x<0 else x)
    pv_est = np.sum(rad["PV"])
    
    # ratio = np.round(pv_ken/pv_est,4)
    # print(mm,pv_ken,pv_est,"ratio=",np.round(pv_ken/pv_est,2))
    return pv_ken, pv_est
  
  #-----------
  
  _ratio = []
  for mm in _mm:
    if cate=="T1":
      pv_ken, pv_est = read_result(mm,p=None) #p0.0-> 44地点全地点(昨年データをすべて利用)
      # print(mm,ratio)
    elif cate=="Z1":
      pv_ken, pv_est = calc_PV(cate,mm)
    elif cate=="Z2":
      pv_ken0, pv_est0 = read_result(mm,p=None) #p0.0-> 44地点全地点(昨年データをすべて利用)
      pv_ken1, pv_est1 = calc_PV(cate,mm)
      pv_ken = pv_ken0 + pv_ken1 #　テレメ+全量(smame)
      pv_est = pv_est0 + pv_est1 #　テレメ+全量(smame)
      
    ratio = pv_ken / pv_est
    _ratio.append(ratio)
  
  df = pd.DataFrame()
  df["mm"] = _mm
  df["ratio"] = _ratio
  df.to_csv(f"../../tbl/teleme/hosei/hosei_{cate}.csv", index=False)
  print(datetime.now(),"[END]", f"../../tbl/teleme/hosei/hosei_{cate}.csv")
  return 

if __name__ == "__main__":
  
  if 1:
    # -----
    _cate=["T1","Z1","Z2"]
      
    for cate in _cate:
      reg_hosei_ratio(cate)
      # sys.exit()