# -*- coding: utf-8 -*-
# when   : 2021.09.15 "時刻別回帰の回帰係数の策定"
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
from getErrorValues import me,rmse,mae,r2,mape #(x,y)
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

from utils_data import reg_use_data

from mapbox import map_lonlat3 #(df,html_path,zoom=4,size_max=None)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures


from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 14
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']


from utils_teleme import *
from mapbox import map_lonlat3 #(df,html_path,zoom=4,size_max=None)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

from utils_plotly import plotly_2axis #(df,col1,col2,html_path, title="sampe"):
from utils_plotly import plotly_1axis #(df,_col,html_path,title="sampe",vmax=1000)
TELEME="/work/ysorimachi/hokuriku/dat2/teleme/dataset" #30min_201912.csv
ESTIMATE2="/home/ysorimachi/work/hokuriku/out/teleme/pu/estimate2"

def load_PV(cate="train",mm=None):
  path = f"{ESTIMATE2}/PV_{cate}_teleme_8now0.csv"
  # print("load_PV->",path)
  df = pd.read_csv(path)
  df["time"] = pd.to_datetime(df["time"])
  df["mm"] = df["time"].apply(lambda x: x.strftime("%Y%m"))
  df = df.set_index("time")
  if mm:
    df = df[df["mm"]==mm]
  return df

def plot_scatter(df,tc="sum",_pc=["PV-max_M1_SN0"]):
  f,ax = plt.subplots(figsize=(15,15))
  for pc in _pc:
    ax.scatter(df[tc],df[pc], label=pc, s=1)
    
  vmax = int(np.max(df[tc]))
  ax.set_title(f"N={df.shape[0]}")
  ax.legend()
  ax.plot(np.arange(vmax),np.arange(vmax), lw=1,color="k")
  f.savefig("./tmp.png",bbox_inches="tight")
  return 


def check_me(cate="train"):
  df = load_PV(cate=cate,mm=None)
  
  print(df.head())
  print(df.columns)
  _pc = ['PV-max[1]','PV-max_M1_SN0','PV-max_M2_SN0']
  plot_scatter(df,tc="sum",_pc=_pc)
  sys.exit()
  
def check_reg(mm="201905"):
  OUTDIR = "/home/ysorimachi/work/hokuriku/out/teleme/pu/tmp"
  if int(mm)<=202003:
    cate="train"
  else:
    cate="test"
  
  """ 
  2021.09.02 
  2021.09.08 with- smame も実装済 
  
  """
  basesize = 5
  f,ax = plt.subplots(1,3,figsize=(basesize*3,basesize*1))
  # ax = ax.flatten()
  # _mm19 = loop_month(st="201904")[:12]
  rcParams['font.size'] = 24
  
  # line
  X_line = np.linspace(0,1,1000)
  _cate2 = ["T1","Z1","Z2"]
  # _cate2 = ["Z1"]
  hh ="1200"
  
  for i ,cate2 in enumerate(_cate2):
    a,b,c = load_mm_hh_abc2(mm,hh,cate=cate,cate2=cate2)
    df = reg_use_data(mm=mm,hh=hh,radname="8now0",select="all",cate2=cate2)
    # print(a,b,c)
    # print(df.head())
    # continue
    # sys.exit()
    # sys.exit()
    
    pu_pred0 = a*X_line**2 + b*X_line + c
    
    df = df[["obs","p.u"]].dropna()
    # print(df.head())
    # sys.exit()
    r2score = np.round(r2_score(df["obs"],df["p.u"]),3)
    
    label = f"R2_score={r2score}"
    ax[i].plot(X_line,pu_pred0,lw=5,color="r", label=label)
    ax[i].scatter(df["obs"],df["p.u"], s=10)
    # pu_pred1 = a1*X_line**2 + b1*X_line + c1
    # pu_pred2 = lr.predict(pf.fit_transform(X_line.reshape(-1,1)))
  # if 1:
  #   ax[i].plot(X_line,pu_pred0,lw=1,color="k", label="201905")
  #   ax[i].plot(X_line,pu_pred1,lw=1,color="b",label="per-Month")
  #   ax[i].plot(X_line,pu_pred2,lw=1,color="r", label="2020-ver")
  
    ax[i].legend(loc="upper left",fontsize=15)
    ax[i].set_title(f"{cate2}")
    
    ax[i].set_ylim(0,1)
    ax[i].set_xlim(0,1)
    ax[i].set_ylabel("p.u.")
    ax[i].set_xlabel("エリア平均日射量[kW/m2]")
  print(datetime.now(),"[end]", mm)
    # sys.exit()
    #---------------------------------------------------------
  # plt.subplots_adjust(wspace=0.4, hspace=0.4)
  plt.subplots_adjust(wspace=0.2, hspace=0.2)
  f.savefig(f"{OUTDIR}/check_reg.png",bbox_inches="tight")
  print(OUTDIR)
  return 
  
  

if __name__ == "__main__":
  if 1:
    """
    2021.10.03 調査を行う予定
    """
    # plot_eda_outlier(mm="202101")
    # plot_teleme_dd(dd="20210113")
    
    # plot_indivisual_dd(code="telm014",dd="20210112")
    # plot_indivisual_rad(code="telm014",dd="20210112")
    # check_me()
    check_reg(mm="201905")