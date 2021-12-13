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

import matplotlib.colors as mcolors
_color = [ mcolors.rgb2hex(plt.cm.get_cmap('tab10')(i)) for i in range(10)]
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 18
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
from utils_teleme import *
sys.path.append("..")
from utils import *
from mapbox import map_lonlat3 #(df,html_path,zoom=4,size_max=None)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from plot1m import plot1m#(df,_col,vmin=0,vmax=1000,month=None,step=None,figtype="plot",title=False):

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
import matplotlib.colors as mcolors
_color = [ mcolors.rgb2hex(plt.cm.get_cmap('tab10')(i)) for i in range(10)]

TELEME="/work/ysorimachi/hokuriku/dat2/teleme/dataset" #30min_201912.csv

# ESTIMATE="/home/ysorimachi/work/hokuriku/out/teleme/pu/estimate"
ESTIMATE="/home/ysorimachi/work/hokuriku/out/teleme/pu/estimate2"
# ERR="/home/ysorimachi/work/hokuriku/out/teleme/pu/err"
ERR="/home/ysorimachi/work/hokuriku/out/teleme/pu/err2"

def load_teleme(month,min=30):
  path = f"{TELEME}/{min}min_{month}.csv"
  df = pd.read_csv(path)
  df["time"] = pd.to_datetime(df["time"])
  df = df.set_index("time")
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

def err_model(err_name):
  if err_name =="rmse":
    return rmse
  if err_name == "me":
    return me
  if err_name == "%mae":
    return mape

# def scatter_PV(cate="mm",radname="8now0",p=None,yy=2019):
def scatter_PV(cate="train",radname="8now0",with_smame = True, winter=False,CI=False):
  """
  2021.10.03 :　月ごと/時刻毎のPV出力表示
  2021.12.08 :　月ごと/時刻毎のPV出力表示
  """
  if cate=="train":
    _mm = loop_month()[:12]
  else:
    _mm = loop_month()[12:24]
    
  def cut_hh(df,st,ed):
    df = df[(df["hh"]>=st)&(df["hh"]<=ed)]
    return df
  
  f,ax = plt.subplots(3,4,figsize=(4*5,3*5))
  rcParams['font.size'] = 24
  # title_size=36
  rcParams['ytick.labelsize'] = 24
  rcParams['xtick.labelsize'] = 24
  # figsize=tuple([18,10])
  # print(rcParams)
  # sys.exit()
  
  
  ax = ax.flatten()
  
  # _point,cN = get_profile(p)
      
  OUTDIR="/home/ysorimachi/work/hokuriku/out/teleme/pu/png"
  # ERR="/home/ysorimachi/work/hokuriku/out/teleme/pu/err"
  # ESTIMATE="/home/ysorimachi/work/hokuriku/out/teleme/pu/estimate"
  # err_hash ={}
  
  # _df=[ pd.read_csv(f"{ESTIMATE}/{mm}_{radname}_{cN}.csv") for mm in _mm ]
  # df_hh = pd.concat(_df,axis=0)
  if with_smame:
    estimate_path = f"{ESTIMATE}/PV_{cate}_teleme_{radname}_sm.csv"
  else:
    estimate_path = f"{ESTIMATE}/PV_{cate}_teleme_{radname}.csv"
  df0 = pd.read_csv(estimate_path)

  df0["time"] = pd.to_datetime(df0["time"])
  df0["hh"] = df0["time"].apply(lambda x: x.hour)
  df0["mm"] = df0["time"].apply(lambda x: x.strftime("%Y%m"))
  df0 = cut_hh(df0,st=6,ed=18)
  _hh=list(range(6,6+12))
  vmax = np.ceil(df0.describe().T["max"].max()/1000*1.05) #2021.10.03 数字切り上げ
  # print(cate,radname,p,yy,"vmax=",vmax)
  # sys.exit()
  # print(df.head())
  # sys.exit()
  
  # print(_mm)
  # print(df0.head())
  # sys.exit()
  #-月別(12)----------------------------------
  # for i,mm in enumerate(_mm):
    # df = df0[df0["mm"]==mm]
    # title = f"{mm}-PV"
  #-時刻別(12)----------------------------------
  for i,hh in enumerate(_hh):
    df = df0[df0["hh"]==hh]
    # title = f"時刻{hh}時台-PV"
    title = f"{hh}時台"
  #-----------------------------------
    # pv_col = [ c for c in df.columns if "-max" in c or "-panel" in c]
    # pv_col = [ c for c in df.columns if "-max" in c]
    pv_col = [ f"PV-max[{j}]" for j in [1,3,4]] #2021.12.08 -----
    
    df = df[["sum"] + pv_col]
    
    legend_hash={
      # "PV-max[0]" : "2019/05(obs)",
      "PV-max[1]" : "ベンチマーク手法",
      # "PV-max[2]" : "月別(8Now0)",
      "PV-max[3]" : "時刻別(8Now0)",
      "PV-max[4]" : "月別/時刻別(8Now0)",
    }
    # legend_hash2=["手法1","手法3","手法4"]
    if 1:
      # kwh -> 10^3 * kWh(MWh)
      df/=1000
    df= df.dropna()
    # print(df.describe())
    # sys.exit()
    # vmax = 60
    # print(vmax)
    # sys.exit()
    for j,c in enumerate(pv_col):
      # print(c)
      r2_v = np.round(r2_score(df["sum"],df[c]),3)
      # print(r2_v)
      # sys.exit()
      #label
      # lbl = legend_hash[c]
      # lbl = f"{legend_hash2[j]}R2={str(r2_v)}"
      lbl = f"{legend_hash[c]}"
      
      marker_size=5
      mr = 5 if j==0 else 1
      mr_alpha = 1 if j==0 else 0.3
      
      ax[i].scatter(df["sum"],df[c],label=lbl,s = marker_size*mr, alpha=mr_alpha)
      
        
    ax[i].set_ylim(0,vmax)
    ax[i].set_xlim(0,vmax)
    ax[i].plot(np.arange(vmax),np.arange(vmax),lw=1,color="k")
    
    # plt.rcParams['text.usetex'] = True
    # xlbl,ylbl = fr"双方向端末+全量スマメPV[×$10^3$kWh]",r"推定PV[×$10^3$kWh]"
    # xlbl,ylbl = fr"双方向端末+全量スマメPV[×MWh]",r"推定PV[×MWh]"
    xlbl,ylbl = fr"合算PV出力[×MWh]",r"推定PV[×MW]"
    # ax[i].set_ylabel(ylbl,fontsize=24)
    # ax[i].set_xlabel(xlbl,fontsize=24)
    #---
    # if i==3:
    #   ax[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    if 0:
      ax[i].legend(fontsize=10)
    
    ax[i].set_title(title)
    
  plt.subplots_adjust(wspace=0.3, hspace=0.3)
  f.savefig(f"{OUTDIR}/scatter_{cate}_PV_{radname}.png",bbox_inches="tight")
  print(f"{OUTDIR}/scatter_{cate}_PV_{radname}.png")
  return 

def scatter_PV2(p="all",method=3):
  """
  2021.10.03 :　月ごと/時刻毎のPV出力表示
  """
  radname = "8now0"
  train_mm = loop_month(st="201904")[:12]
  test_mm = loop_month(st="201904")[12:]
  
  f,ax = plt.subplots(3,4,figsize=(4*5,3*4))
  ax = ax.flatten()
  _point,cN = get_profile(p)
      
  OUTDIR="/home/ysorimachi/work/hokuriku/out/teleme/pu/png"
  # ERR="/home/ysorimachi/work/hokuriku/out/teleme/pu/err"
  # ESTIMATE="/home/ysorimachi/work/hokuriku/out/teleme/pu/estimate"
  # err_hash ={}
  
  _train=[ pd.read_csv(f"{ESTIMATE}/{mm}_{radname}_{cN}.csv") for mm in train_mm ]
  _test=[ pd.read_csv(f"{ESTIMATE}/{mm}_{radname}_{cN}.csv") for mm in test_mm ]
  
  _df2 = []
  vmax = 75
  
  rcParams['font.size'] = 12
  legend_hash={
    "PV-max[0]" : "2019/05(obs)",
    "PV-max[1]" : "2019/05(obs)+Hosei",
    "PV-max[2]" : "月別(8Now0)",
    "PV-max[3]" : "時刻別(8Now0)",
  }
  for i,_df in enumerate([_train,_test]):
    
    df = pd.concat(_df,axis=0)
    df["time"] = pd.to_datetime(df["time"])
    df["hh"] = df["time"].apply(lambda x: x.hour)
    vmax = np.ceil(df.describe().T["max"].max()/1000*1.05) #2021.10.03 数字切り上げ
    _df2.append(df)
  #------------------
  _hh=list(range(6,6+12))
  _yy=[2019,2020]
  _color = ["b","r"]
  pv_col = f"PV-max[{method}]"
  for i in range(12):
    # path = f"{ESTIMATE}/{mm}.csv"
    hh = _hh[i]
    
    for df,yy,color in zip(_df2,_yy,_color):
      
      tmp= df[df["hh"]==hh] #train 2019
      tmp = tmp[["sum",pv_col]]
      tmp = tmp.dropna()
    
      if 1:
        # kwh -> 10^3 * kWh
        tmp/=1000
      # sys.exit()
      r2_v = np.round(r2_score(tmp["sum"],tmp[pv_col]),3)
      
      lbl = f"{yy} R2={str(r2_v)}"
      ax[i].scatter(tmp["sum"],tmp[pv_col],label=lbl,s=1.5,color=color)
      
        
    ax[i].set_ylim(0,vmax)
    ax[i].set_xlim(0,vmax)
    ax[i].plot(np.arange(100),np.arange(100),lw=1,color="k")
    
    # plt.rcParams['text.usetex'] = True
    xlbl,ylbl = fr"双方向端末PV({cN})[×$10^3$kWh]",r"推定PV[×$10^3$kWh]"
    ax[i].set_ylabel(ylbl,fontsize=12)
    ax[i].set_xlabel(xlbl,fontsize=12)
    #---
    # if i==3:
    #   ax[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    if 1:
      ax[i].legend()
    
    title=f"{hh}時台"
    ax[i].set_title(title)
  plt.subplots_adjust(wspace=0.4, hspace=0.4)
  f.savefig(f"{OUTDIR}/scatter_ratioPV_{radname}_{cN}.png",bbox_inches="tight")
  
  


def plot_ts_mm(mm="202005",radname="8now0",p=None):
  _point,cN = get_profile(p)
  
  df = pd.read_csv(f"{ESTIMATE}/{mm}_{radname}_{cN}.csv")
  legend_hash={
    "PV-max[0]" : "2019/05(obs)",
    "PV-max[1]" : "2019/05(obs)+Hosei",
    "PV-max[2]" : "月別(8Now0)",
  }
  df = df.rename(columns = legend_hash)
  plot_col = ["sum",'2019/05(obs)', '2019/05(obs)+Hosei', '月別(8Now0)']
  vmax = df[plot_col].describe().T["max"].max()/1000*1.05
  
  for c in plot_col:
    df[c] /=1000
  
  f = plot1m(df,_col=["sum",'2019/05(obs)', '2019/05(obs)+Hosei', '月別(8Now0)'],vmin=0,vmax=vmax,month=mm,step=3,figtype="plot",title=None)
  PNG_DIR="/home/ysorimachi/work/hokuriku/out/teleme/pu/png"
  f.savefig(f"{PNG_DIR}/ts_{mm}_{cN}.png", bbox_inches="tight")
  print(df.head())
  sys.exit()

def get_profile(p):
  if p == "all":
    # _point = sorted(select_teleme(threshold=0))
    _point = ["all"]
    cN="all"
  elif type(p) == int:
    _point = sorted(select_teleme(threshold=p))
    cN=f"sel{p}%"
  else:
    # _point = sorted(sonzai_sort(N=5))
    _point = ['telm032', 'telm016', 'telm030', 'telm034', 'telm037'] #  #debug -- 21.10.01
    cN="sel_SORI"
  return _point,cN

def bar_seido_hh(err_name="rmse",cate="train",with_smame=False, winter=False, CI=False):
  
  ODIR="/home/ysorimachi/work/hokuriku/out/teleme/pu/png_err_bar"
  _point,cN  =get_profile(p)
  radname = "8now0"
  # path = f"{ERR}/{cate}_{yy}_teleme_{radname}_{cN}.csv"
  add="_sm" if with_smame else ""
  cwi=1 if winter else 0
  cci=1 if CI else 0
    
  # path = f"{ERR}/seido_hh_{cate}_teleme_{radname}{add}.csv" #path 変更 2021.11.19
  if with_smame:
    path = f"{ERR}/seido_hh_{cate}_teleme_{radname}{add}.csv"
  else:
    path = f"{ERR}/seido_hh_{cate}_teleme_{radname}_wi{cwi}_CI{cci}{add}.csv" #path 変更 2021.11.19
  # xlbl,ylbl = "時刻",fr"{err_name}[×$10^3$kWh]"
  print(path)
  xlbl,ylbl = "時刻","[×MW]"
  
  if err_name[0] =="n":
    ylbl = fr"{err_name}[-]"
  
  p_rate =1 if p=="select" else 10 
  if err_name=="me":
    vmin,vmax = -500*p_rate,500*p_rate
  if err_name=="rmse":
    vmin,vmax = 0,800*p_rate
  if err_name=="nrmse":
    vmin,vmax = 0,0.5
  
  def set_columns(df,err_name):
    df = df.set_index("hh")
    _col = [ c for c in df.columns if c.startswith(err_name)]
    df =df[_col]
    # df.columns = names
    # df = df.iloc[4:29,:]
    return df

  def kaizen_RMSE(df,c0,c1,method="ratio"):
    if method == "ratio":
      # for c in df.columns:
      df[c1] =  -1 * (df[c1]-df[c0]) * 100 / df[c0]
    if method=="diff":
      # for c in df.columns:
      df[c1] =  1 * (df[c1] - df["base"])
    
    # df = df.round(2)
    # df = df.drop(["base"], axis=1)
    return df
  
  def kaizen_ME(df,c0,c1,method="ratio"):
    if method == "ratio":
      # for c in df.columns:
      df[c1] =  -1 * (np.abs(df[c1])-np.abs(df[c0])) * 100 / df[c0]
    return df
  
  def bar_plot(df,path, isRatio=True,bar_alpha=1,v_range=[0,1000],figsize=(18,8)):
    f,ax = plt.subplots(figsize=figsize)
    # w=0.2 n_columns = 4
    vmin,vmax = v_range
    w = 1/(df.shape[1]+1)
    _index=df.index
    for i,c in enumerate(df.columns):
      color = _color[i]
      ax.bar(w/2 + np.arange(len(df))+w*i,df[c],width=w,label=c,align="edge", alpha=bar_alpha,color=color)
    
    
    #------------
    #-------------
    if isRatio:
      _c= df.columns[1:]
      if err_name == "rmse":
        pmin,pmax = -100,100
        for c in _c:
          df = kaizen_RMSE(df,c0="ベンチマーク手法",c1=c,method="ratio")
          
      if err_name == "me":
        pmin,pmax = -200,200
        for c in _c:
          df = kaizen_ME(df,c0="ベンチマーク手法",c1=c,method="ratio")
    
    #------------------------
    if isRatio:
      bx  = ax.twinx()
      bx.set_ylim(pmin,pmax)
      bx.set_ylabel("改善率[%](+:改善/-:悪化)")
      for i,c in enumerate(df.columns):
        color = _color[i]
        if i>0:
          bx.plot(w/2 + np.arange(len(df))+w*i,df[c],marker="o",lw=1,color=color)
        bx.axhline(y=1,lw=1, color="gray", alpha=0.4)
    #------------------------
    #-----------------------------------
    # ax.grid()
    # ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    ax.set_ylim(vmin,vmax)
    ax.set_xlim(0,len(df))
    for x0 in np.arange(len(df)):
      ax.axvline(x=x0,color="gray", alpha=0.4, lw=1)
    if err_name =="me":
      ax.axhline(y=0,color="k", lw=1)
    ax.set_xticks(np.arange(len(df))+1/2)
    ax.set_xticklabels(_index,rotation=90)
    
    # ax.legend(loc = "upper left")
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    # ax.legend(loc='upper right')
    ax.set_title(f"{err_name}({cate})")
    f.savefig(path,bbox_inches="tight")
    return 
  #load df ---
  df = pd.read_csv(path)
  df = set_columns(df,err_name)
  df = df.dropna()
  
  #----with - sumame -----
  # use_col = [f"{err_name}(1)",f"{err_name}(3)",f"{err_name}(4)"]
  # renames = ["ベンチマーク手法",f"時刻別",f"月別・時刻別"]
  #---- 過積載の影響  
  # use_col = [f"{err_name}(1)",f"{err_name}(5)",f"{err_name}(7)"]
  # renames = ["ベンチマーク手法",f"設備量メッシュ(-)",f"設備量メッシュ(過積載)"]
  #---- 積雪補正の影響  
  use_col = [f"{err_name}(1)",f"{err_name}(7)",f"{err_name}(8)"]
  renames = ["ベンチマーク手法",f"設備量メッシュ(過積載)",f"設備量メッシュ(過積載+積雪)"]
  #---- default 
  # use_col = [ f"{err_name}({i})" for i in [0,1,2,3,4]]
  # renames = ["手法1:2019年5月の特性曲線","手法2: ベンチマーク","手法3: 月別","手法4: 時刻別","手法5: 月別/時刻別"]
  #----with - sumame -----
  # print(df.head())
  # sys.exit()
  
  df = df[use_col]
  df.columns = renames
  # print(df.head())
  # sys.exit()
  try:
    df = df.drop("18:30")
  except:
    pass
  # print(df.tail())
  # sys.exit()
  #save bar ---
  rcParams['font.size'] = 36
  rcParams['ytick.labelsize'] = 36
  rcParams['xtick.labelsize'] = 24
  figsize=tuple([18,10])
  # print(rcParams)
  # sys.exit()
  
  df /=1000 # --> MW
  path = f"{ODIR}/bar_hh_{cate}_{err_name}_teleme.png"
  
  
  if err_name=="me":
    if with_smame:
      v_range=[-5,5]
    else:
      # v_range=[-2000,2000]
      # v_range=[-2,2]
      if CI:
        v_range=[-4,4]
        figsize=tuple([18,8])
      else:
        v_range=[-2,2]
  if err_name=="rmse":
    if with_smame:
      v_range=[0,8]
    else:
      if CI:
        v_range=[0,4]
        figsize=tuple([18,8])
      else:
        v_range=[0,4]
    
  bar_plot(df,path, isRatio=False,bar_alpha=1,v_range=v_range,figsize=figsize)
  #log
  print(ODIR, cate, err_name)
  return

def bar_seido_hh_setsubi(err_name="rmse",cate="train",winter=False, CI=False):
  
  ODIR="/home/ysorimachi/work/hokuriku/out/teleme/pu/png_err_bar"
  _point,cN  =get_profile(p)
  radname = "8now0"
  # path = f"{ERR}/{cate}_{yy}_teleme_{radname}_{cN}.csv"
  # add="_sm" if with_smame else ""
  cwi=1 if winter else 0
  cci=1 if CI else 0
    
  # path = f"{ERR}/seido_hh_{cate}_teleme_{radname}{add}.csv" #path 変更 2021.11.19
  path0 = f"{ERR}/seido_hh_{cate}_teleme_8now0_sm.csv"
  path1 = f"{ERR}/seido_hh_{cate}_teleme_8now0_ind.csv" #path 変更 2021.11.19
  # xlbl,ylbl = "時刻",fr"{err_name}[×$10^3$kWh]"
  df0 = pd.read_csv(path0).set_index("hh")
  df1 = pd.read_csv(path1).set_index("hh")
  
  c=f"{err_name}(4)"
  df = pd.concat([df0[f"{err_name}(1)"],df0[c],df1[c]],axis=1)
  # print(df.head(20))
  # sys.exit()
  
  xlbl,ylbl = "時刻","[×MW]"
  
  if err_name[0] =="n":
    ylbl = fr"{err_name}[-]"
  
  p_rate =1 if p=="select" else 10 
  if err_name=="me":
    vmin,vmax = -500*p_rate,500*p_rate
  if err_name=="rmse":
    vmin,vmax = 0,800*p_rate
  if err_name=="nrmse":
    vmin,vmax = 0,0.5
  
  def set_columns(df,err_name):
    df = df.set_index("hh")
    _col = [ c for c in df.columns if c.startswith(err_name)]
    df =df[_col]
    # df.columns = names
    # df = df.iloc[4:29,:]
    return df

  def kaizen_RMSE(df,c0,c1,method="ratio"):
    if method == "ratio":
      # for c in df.columns:
      df[c1] =  -1 * (df[c1]-df[c0]) * 100 / df[c0]
    if method=="diff":
      # for c in df.columns:
      df[c1] =  1 * (df[c1] - df["base"])
    
    # df = df.round(2)
    # df = df.drop(["base"], axis=1)
    return df
  
  def kaizen_ME(df,c0,c1,method="ratio"):
    if method == "ratio":
      # for c in df.columns:
      df[c1] =  -1 * (np.abs(df[c1])-np.abs(df[c0])) * 100 / df[c0]
    return df
  
  def bar_plot(df,path, isRatio=True,bar_alpha=1,v_range=[0,1000],figsize=(18,8)):
    f,ax = plt.subplots(figsize=figsize)
    # w=0.2 n_columns = 4
    vmin,vmax = v_range
    w = 1/(df.shape[1]+1)
    _index=df.index
    for i,c in enumerate(df.columns):
      color = _color[i]
      ax.bar(w/2 + np.arange(len(df))+w*i,df[c],width=w,label=c,align="edge", alpha=bar_alpha,color=color)
    
    
    #------------
    #-------------
    if isRatio:
      _c= df.columns[1:]
      if err_name == "rmse":
        pmin,pmax = -100,100
        for c in _c:
          df = kaizen_RMSE(df,c0="ベンチマーク手法",c1=c,method="ratio")
          
      if err_name == "me":
        pmin,pmax = -200,200
        for c in _c:
          df = kaizen_ME(df,c0="ベンチマーク手法",c1=c,method="ratio")
    
    #------------------------
    if isRatio:
      bx  = ax.twinx()
      bx.set_ylim(pmin,pmax)
      bx.set_ylabel("改善率[%](+:改善/-:悪化)")
      for i,c in enumerate(df.columns):
        color = _color[i]
        if i>0:
          bx.plot(w/2 + np.arange(len(df))+w*i,df[c],marker="o",lw=1,color=color)
        bx.axhline(y=1,lw=1, color="gray", alpha=0.4)
    #------------------------
    #-----------------------------------
    # ax.grid()
    # ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    ax.set_ylim(vmin,vmax)
    ax.set_xlim(0,len(df))
    for x0 in np.arange(len(df)):
      ax.axvline(x=x0,color="gray", alpha=0.4, lw=1)
    if err_name =="me":
      ax.axhline(y=0,color="k", lw=1)
    ax.set_xticks(np.arange(len(df))+1/2)
    ax.set_xticklabels(_index,rotation=90)
    
    # ax.legend(loc = "upper left")
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    # ax.legend(loc='upper right')
    ax.set_title(f"{err_name}({cate})")
    f.savefig(path,bbox_inches="tight")
    return 
  
  #load df ---
  # df = pd.read_csv(path)
  # df = set_columns(df,err_name)
  # df = df.dropna()
  
  #----with - sumame -----
  # use_col = [f"{err_name}(1)",f"{err_name}(3)",f"{err_name}(4)"]
  # renames = ["ベンチマーク手法",f"時刻別",f"月別・時刻別"]
  #---- 過積載の影響  
  use_col = [f"{err_name}(1)",f"{err_name}(5)",f"{err_name}(7)"]
  renames = ["ベンチマーク手法",f"設備量メッシュ(-)",f"設備量メッシュ(過積載)"]
  #---- 積雪補正の影響  
  use_col = [f"{err_name}(1)",f"{err_name}(7)",f"{err_name}(8)"]
  renames = ["ベンチマーク手法",f"設備量メッシュ(過積載)",f"設備量メッシュ(過積載+積雪)"]
  #---- 積雪補正の影響  
  # use_col = [ f"{err_name}({i})" for i in [0,1,2,3,4]]
  # renames = ["手法1:2019年5月の特性曲線","手法2: ベンチマーク","手法3: 月別","手法4: 時刻別","手法5: 月別/時刻別"]
  #----　設備ごとの誤差評価
  use_col = [f"{err_name}(1)",f"{err_name}(4)",f"{err_name}(4)"]
  renames = ["ベンチマーク手法",f"合算(全量)",f"個別(teleme/smame)"]
  
  df = df.dropna()
  # df = df[use_col]
  df.columns = renames
  # print(df.head())
  # sys.exit()
  try:
    df = df.drop("18:30")
  except:
    pass
  # print(df.tail())
  # sys.exit()
  # print(df.head())
  # sys.exit()
  rcParams['font.size'] = 36
  rcParams['ytick.labelsize'] = 36
  rcParams['xtick.labelsize'] = 24
  figsize=tuple([18,10])
  # print(rcParams)
  # sys.exit()
  
  df /=1000 # --> MW
  #save bar ---
  path = f"{ODIR}/bar_hh_{cate}_{err_name}_setsubi.png"
  if err_name=="me":
    v_range=[-5,5]
  if err_name=="rmse":
    v_range=[0,8]
    
  bar_plot(df,path, isRatio=False,bar_alpha=1,v_range=v_range,figsize=figsize)
  #log
  print(ODIR, cate, err_name)
  return



def bar_seido_mm(err_name="rmse",cate="train",with_smame=False):
  """
  2021.10.11 記述
  2021.11.19 記述
  2021.12.07 記述
  """
  
  ODIR="/home/ysorimachi/work/hokuriku/out/teleme/pu/png_err_bar"
  _point,cN  =get_profile(p)
  radname = "8now0"
  # path = f"{ERR}/{cate}_{yy}_teleme_{radname}_{cN}.csv"
  # path = f"{ERR}/only_teleme_{radname}_{cN}.csv"
  if with_smame:
    add="_sm"
  else:
    add=""
  path = f"{ERR}/seido_mm_{cate}_teleme_{radname}{add}.csv" #path 変更 2021.11.19
  # path = f"{ERR}/seido_mm_{cate}_teleme_{radname}.csv" #path 変更 2021.11.19
  # xlbl,ylbl = "時刻",fr"{err_name}[×$10^3$kWh]"
  xlbl,ylbl = "月(YYYYmm)",fr"{err_name}[×MWh]"
  
  if err_name[0] =="n":
    ylbl = fr"{err_name}[-]"
  
  p_rate =1 if p=="select" else 10 
  if err_name=="me":
    vmin,vmax = -500*p_rate,500*p_rate
  if err_name=="rmse":
    vmin,vmax = 0,800*p_rate
  if err_name=="nrmse":
    vmin,vmax = 0,1
  # ------------------------------------------------------------------
  def set_columns(df,err_name):
    names = [ 
             "手法1 : 2019/05(obs)",
             "手法2 : 2019/05(obs)+Hosei",
             "手法3 : 月[mm](8Now0)",
             "手法4 : 時刻[hh](8Now0)",
             "手法5 : 月[mm]-時刻[hh](8Now0)",
             "手法6 : Mesh(max)-Snow(なし)",
             "手法7 : Mesh(over)-Snow(なし)",
             "手法8 : Mesh(max)-Snow(あり)",
             "手法9 : Mesh(over)-Snow(あり)",
    ]
    # df = df.set_index(cate)
    df = df.set_index("month")
    _col = [ c for c in df.columns if c.startswith(err_name)]
    # print(_col)
    # sys.exit()
    df =df[_col]
    # df.columns = names
    # df = df.iloc[4:29,:]
    return df
  
  def bar_plot(df,path):
    f,ax = plt.subplots(figsize=(18,8))
    # w=0.2 n_columns = 4
    w = 1/(df.shape[1]+1)
    # print(df.shape)
    # sys.exit()
    
    _index=df.index
    for i,c in enumerate(df.columns):
      ax.bar(w/2 + np.arange(len(df))+w*i,df[c],width=w,label=c,align="edge")
    
    
    # ax.grid()
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    ax.set_ylim(vmin,vmax)
    ax.set_xlim(0,len(df))
    for x0 in np.arange(len(df)):
      ax.axvline(x=x0,color="gray", alpha=0.7, lw=1)
    if err_name =="me":
      ax.axhline(y=0,color="k", lw=1)
    ax.set_xticks(np.arange(len(df))+1/2)
    # ax.set_xticklabels(_index,rotation=80)
    ax.set_xticklabels(_index,rotation=0)
    
    # ax.legend(loc = "upper left")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    # ax.set_title(f"{err_name}({yy})-N[{p}]")
    ax.set_title(f"{err_name}({cate})")
    f.savefig(path,bbox_inches="tight")
    return 
  # ------------------------------------------------------------------
  #load df -----------------------------------
  df = pd.read_csv(path)  
  df = set_columns(df,err_name)
  
  # use_col = [ f"{err_name}({i})" for i in [0,1,2,3,4]]
  # names = [
  #   "手法1:2019年5月の特性曲線",
  #   "手法2: ベンチマーク",
  #   "手法3: 月別",
  #   "手法4: 時刻別",
  #   "手法5: 月別/時刻別",
  #   ]
  #---- 過積載の影響  
  # use_col = [f"{err_name}(1)",f"{err_name}(5)",f"{err_name}(7)"]
  # names = ["ベンチマーク手法",f"設備量メッシュ(-)",f"設備量メッシュ(過積載)"]
  #---- 積雪補正の影響  
  use_col = [f"{err_name}(1)",f"{err_name}(7)",f"{err_name}(8)"]
  names = ["ベンチマーク手法",f"設備量メッシュ(過積載)",f"設備量メッシュ(過積載+積雪)"]
  
  df = df[use_col]
  df.columns = names
  # print(df.head())
  # sys.exit()
  
  #save bar ---
  path = f"{ODIR}/bar_mm_{cate}_{err_name}_teleme.png"
  bar_plot(df,path)
  #log
  print(cate,err_name,"ODIR->",ODIR)
  # print("ODIR->",ODIR)
  return



def kaizen_ratio(err_name="rmse",yy=2019,cate="hh",p="all",method="ratio",base_col="手法2"):
  ODIR="/home/ysorimachi/work/hokuriku/out/teleme/pu/png_err_bar"
  _point,cN  =get_profile(p)
  radname = "8now0"
  path = f"{ERR}/{cate}_{yy}_teleme_{radname}_{cN}.csv"
  xlbl,ylbl = "時刻",fr"{err_name}[×$10^3$kWh]"
  
  def set_columns(df,err_name):
    names = [ "手法1", "手法2","手法3","手法4"]
    df = df.set_index(cate)
    _col = [ c for c in df.columns if c.startswith(err_name)]
    df =df[_col]
    df.columns = names
    df["base"] = df[base_col]
    df = df.dropna()
    # df = df.iloc[4:29,:]
    return df

  def kaizen(df):
    if method == "ratio":
      for c in df.columns:
        df[c] =  1 * (df[c]-df["base"]) * 100 / df["base"]
    if method=="diff":
      for c in df.columns:
        df[c] =  1 * (df[c] - df["base"])
    
    df = df.round(2)
    df = df.drop(["base"], axis=1)
    return df
      
  #load df ---
  df = pd.read_csv(path)
  df = set_columns(df,err_name)
  df = kaizen(df)
  df = df.drop(["手法1","手法2"], axis=1)
  df.to_csv(f"{ERR}/kaizen_{method}_{cate}_{yy}_{cN}.csv",encoding="shift-jis")
  return
  

if __name__ == "__main__":
  
  # _point = select_teleme(98)
  # print(_point)
  # sys.exit()
  if 1:
    #setting ---
    radname = "8now0"
    p="all"


    for cate in ["train","test"]:
      
      #---
      if 0: #2021.10.15 -> 2021.12.08
        scatter_PV(cate=cate,radname=radname,with_smame = True, winter=False,CI=False)
        
      if 1: #2021.11.25
        for err_name in ["rmse","me"]:
          # bar_seido_mm(err_name = err_name,cate=cate,with_smame=False)
          bar_seido_hh(err_name = err_name,cate=cate,with_smame =False,winter=True,CI=False)#with-smame
          # bar_seido_hh_setsubi(err_name = err_name,cate=cate,winter=False,CI=False)#with-smame
          # sys.exit()
          # sys.exit()
          
          # sys.exit()
      if 0: #2021.10.15
        kaizen_ratio(err_name="rmse",yy=yy,cate=cate,p=p,method="ratio")

  if 0:
    p = "all"
    method =3
    scatter_PV2(p=p,method=method)
    # for err_name in ["rmse","me"]
  
  if 0:
    # Error Plot
    for err_name in ["rmse","me"]:
      seido_plot1(err_name=err_name) #2021.08.11
  
  if 0:
    for radname in ["8now0"]:
      # for p in [None]:
      for p in ["select"]:
        plot_ts_mm(mm="202101",radname=radname,p=p)
        scatter_rad(cate=cate,radname=radname,p=p)