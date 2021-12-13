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

from mapbox import map_lonlat3 #(df,html_path,zoom=4,size_max=None)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures


from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 14
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']


from plot1m import plot1m
TELEME="/work/ysorimachi/hokuriku/dat2/teleme/dataset" #30min_201912.csv
ESTIMATE="/home/ysorimachi/work/hokuriku/out/teleme/pu/estimate2"
PNG="/home/ysorimachi/work/hokuriku/out/teleme/pu/png"
PNG="/home/ysorimachi/work/hokuriku/out/teleme/pu/png2/outliers/teleme"


def load_data(mm):
  path = f"{ESTIMATE}/{mm}_8now0_all.csv"
  df = pd.read_csv(path)
  df["time"] =pd.to_datetime(df["time"])
  df["dd"] = df["time"].apply(lambda x: x.strftime("%Y%m%d"))
  return df


def plot_eda_outlier(mm="202007"):
  """
  日別の誤差を表示してその事例を抽出するようなもの
  """
  out_path = f"{PNG}/ts_{mm}_all.png"
  df = load_data(mm)

  # print(df.columns)
  use_col = ['sum','PV-max[0]', 'PV-max[1]', 'PV-max[2]', 'PV-max[3]']
  for c in use_col:
    df[c] /=1000 #kWh -> 10^3
  use_col2 = ['実績値','手法1', '手法2', '手法3', '手法4']
  renames = { k:v for k,v in zip(use_col,use_col2) }
  df = df.rename(columns = renames)
  # plot -DEBUG --
  if 1:
    f = plot1m(df,_col=use_col2,vmin=0,vmax=50,month=mm,step=2*3,figtype="plot",title=False)
    f.savefig(out_path, bbox_inches="tight")
    print(out_path)
  sys.exit()
  # ---------------
  res = select_outlier(df,_col=["sum","PV-max[2]"], err_name="rmse")
  print(res.head())
  sys.exit()
  
  return


def plot_teleme_dd(dd="20200717"):
  # OUTDIR="/home/ysorimachi/work/hokuriku/out/teleme/pu/png2/outliers/teleme"
  df = load_teleme(dd[:6])
  a,b,c = get_a_b_c(month=dd[:6],cate="8now0")
  
  rad = load_rad(month=dd[:6],cate="8now0", lag=30, only_mean=True)
  rad = rad[rad["dd"]==dd]

  df = df.reset_index()
  df["dd"] = df["time"].apply(lambda x: x.strftime("%Y%m%d"))
  df = df[df["dd"]==dd].set_index("time")
  df = df.drop(["dd"],axis=1)
  
  telm_col = [ c for c in df.columns if "telm" in c]
  
  s = pd.Series(df.isnull().sum()/df.shape[0],name="null_rate")
  
  _code = s.index
  _null = s.values
  f,ax = plt.subplots(7,9,figsize=(25,20))
  ax = ax.flatten()
  
  df = pd.concat([df,rad],axis=1)
  # df = df.rename(columns={"mean": r"平均日射量[W/$m^2$]"})
  rad_x = (df["mean"]/1000)
  for i in tqdm(list(range(63))):
    
    if i<=60:
      code,nan2 = _code[i],_null[i]
      mpv = teleme_max(code=code,cate ="max")
      ppv = teleme_max(code=code,cate ="panel")
      over_rate = np.round((ppv-mpv)*100/mpv,2)
      
      if nan2<0.99:
        ax[i].plot(np.arange(len(df)),df[code].values / mpv, lw=2, color="r", label="実績")
        ax[i].fill_between(np.arange(len(df)),y1=(a*rad_x**2 + b*rad_x + c), lw=1, color="gray", label="推定p.u",alpha=0.5)
        
        # bx.fill_between(np.arange(len(df)),y1=df["mean"]/1000,color="orange", alpha=0.5)
        # ax[i].axhline(y=mpv, color="k",alpha=0.8)
        # ax[i].axhline(y=ppv,color="k", alpha=0.7)
        # bx = ax[i].twinx()
        # bx.fill_between(np.arange(len(df)),y1=df["mean"]/1000,color="orange", alpha=0.5)
        # bx.set_ylim(0,1)
        # bx.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
      else:
        ax[i].text(0.1,0.5 ,"No\n  Record!")
        
      ax[i].set_title(code)
      ax[i].set_ylim(0,1)
      ax[i].tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
      # print(code,np.round(nan2*100,1),"[%]", mpv,ppv,over_rate,"[%]")
      
    else:
      ax[i].set_visible(False)
  
  plt.subplots_adjust(wspace=0.4, hspace=0.5)
  f.savefig(f"{PNG}/outliers_{dd}.png", bbox_inches="tight")
  return 
  
def plot_indivisual_dd(code="telm013",dd="20200717"):
  # OUTDIR="/home/ysorimachi/work/hokuriku/out/teleme/pu/png2/outliers/teleme"
  df = load_teleme(dd[:6])
  a,b,c = get_a_b_c(month=dd[:6],cate="8now0")
  
  rad = load_rad(month=dd[:6],cate="8now0", lag=30, only_mean=True)
  rad = rad[rad["dd"]==dd]

  df = df.reset_index()
  df["dd"] = df["time"].apply(lambda x: x.strftime("%Y%m%d"))
  df = df[df["dd"]==dd].set_index("time")
  df = df.drop(["dd"],axis=1)
  
  telm_col = [ c for c in df.columns if "telm" in c]
  
  s = pd.Series(df.isnull().sum()/df.shape[0],name="null_rate")
  
  _code = s.index
  _null = s.values
  f,ax = plt.subplots(figsize=(13,8))
  # ax = ax.flatten()
  
  df = pd.concat([df,rad],axis=1)
  # df = df.rename(columns={"mean": r"平均日射量[W/$m^2$]"})
  
  mpv = teleme_max(code=code,cate ="max")
  ppv = teleme_max(code=code,cate ="panel")
  #--------cut time (daytime)
  df = df.reset_index()
  df["hh"] = df["time"].apply(lambda x: x.hour)
  df = df[(df["hh"]>=4)&(df["hh"]<21)]
  df = df.set_index("time")
  rad_x = (df["mean"]/1000)
  
  # print(code,ppv,mpv)
  over_rate = np.round(ppv/mpv,4)
  ax.plot(np.arange(len(df)),df[code].values / mpv, lw=2, color="r", label="実績")
  
  ax.fill_between(np.arange(len(df)),y1=(a*rad_x**2 + b*rad_x + c), lw=1, color="gray", label="推定p.u",alpha=0.2)
  ax.fill_between(np.arange(len(df)),y1=(a*rad_x**2 + b*rad_x + c)*over_rate, lw=1, color="purple", label="推定p.u+過積載補正",alpha=0.2)

  _index = [ t.strftime("%H:%M") for t in df.index ] 
  ax.legend()
  ax.set_xlabel("時刻")
  ax.set_xticks(np.arange(len(df)))
  ax.set_xlim(0,len(df))
  ax.set_xticklabels(_index, rotation=80)
  
  ax.set_ylabel(fr"p.u[-]")
  ax.set_title(code)
  ax.set_ylim(0,1)
  
  f.savefig(f"{PNG}/indivisual_{code}_{dd}.png", bbox_inches="tight")
  return 


def plot_indivisual_rad(code="telm013",dd="20200717"):
  # OUTDIR="/home/ysorimachi/work/hokuriku/out/teleme/pu/png2/outliers/teleme"
  rad = load_rad(month=dd[:6],cate="8now0", lag=30, only_mean=False)
  # print(rad.head())
  # sys.exit()
  if code == "telm013":
    code2 = "unyo008"  #telm013
  if code == "telm014":
    code2 = "unyo018"  #telm014
    
  rad = rad [["dd",code2,"mean","std"]]
  rad = rad[rad["dd"]==dd]
  # print(rad.head())
  
  #--------cut time (daytime)
  rad = rad.reset_index()
  rad["hh"] = rad["time"].apply(lambda x: x.hour)
  rad = rad[(rad["hh"]>=4)&(rad["hh"]<21)]
  rad = rad.set_index("time")
  
  f,ax = plt.subplots(figsize=(13,8))
  
  ax.plot(np.arange(len(rad)), rad["mean"].values, label=r"エリア平均日射量[W/$m^2$]")
  ax.fill_between(np.arange(len(rad)), y2 = rad["mean"] - rad["std"],y1 = rad["mean"] + rad["std"], label=r"平均日射量(標準偏差)",alpha=0.5)
  ax.plot(np.arange(len(rad)), rad[code2].values,lw=3,label=f"{code2}({code}近傍地点)")
  
  _index = [ t.strftime("%H:%M") for t in rad.index ] 
  ax.legend()
  ax.set_xlabel("時刻")
  ax.set_xticks(np.arange(len(rad)))
  ax.set_xlim(0,len(rad))
  ax.set_xticklabels(_index, rotation=80)
  
  ax.set_ylabel(r"日射量[W/$m^2$]")
  # ax.set_title(code)
  ax.set_ylim(0,1000)
  
  f.savefig(f"{PNG}/rad_{code}_{dd}.png", bbox_inches="tight")
  plt.close()
  print(PNG)
  return
  

if __name__ == "__main__":
  if 1:
    """
    2021.10.03 調査を行う予定
    """
    # plot_eda_outlier(mm="202101")
    # plot_teleme_dd(dd="20210113")
    
    plot_indivisual_dd(code="telm014",dd="20210112")
    plot_indivisual_rad(code="telm014",dd="20210112")