# -*- coding: utf-8 -*-
# when   : 2021.07.15
# who : [sori-machi]
# what : 北陸技研の記録についてセットアップする(異常値等の確認)
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
from getErrorValues import me,rmse,mape,r2 #(x,y)
from convSokuhouData import conv_sfc #(df, ave=minutes,hour)
#amedas relate 2020.02,04 making...
from tool_AMeDaS import code2name, name2code
from tool_110570 import get_110570,open_110570
from tool_100571 import get_100571,open_100571
#(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
import subprocess

from plot1m import plot1m, plot1m_ec, plot1m_2axis #(df,_col,_mem_col=False,month=False,vmin=0,vmax=1000,title=False)
from plot1d import plot1d_ec #title=False)
sys.path.append("..")
from utils import *
from sklearn.linear_model import LinearRegression

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
import matplotlib.colors as mcolors
_color = [ mcolors.rgb2hex(plt.cm.get_cmap('tab10')(i)) for i in range(10)]

# outd ='/home/griduser/work/sori-py2/deep/out/0924_01'
# os.makedirs(outd, exist_ok=True)
# subprocess.run('rm -f *.png', cwd=outd, shell=True)

from clensing import rad_JOGAI #(df,cate="obs")
from plot_ts import rad_code2name #(code="unyo001")

RAD_DAT="/home/ysorimachi/work/hokuriku/dat/rad/re_get"
OBS_1MIN = "/work/ysorimachi/hokuriku/dat2/rad/obs/1min"
NOW_5MIN = "/work/ysorimachi/hokuriku/dat2/rad/8Now0/5min"

ERR="/home/ysorimachi/work/hokuriku/out/rad/err"
PER_CODE="/work/ysorimachi/hokuriku/dat2/rad/per_code"
OUT_D="/home/ysorimachi/work/hokuriku/dat/rad/r2/per_mm/plot" #2021.11.10
# def clensing_rad(df):
#   df["time"] = pd.to_datetime(df["time"])
  

def load_8now0(month):
  path = f"{NOW_5MIN}/{month}_sat.csv"
  df = pd.read_csv(path)
  df["time"] = pd.to_datetime(df["time"])
  df = rad_JOGAI(df,cate="8now0")
  
  if "unyo000" in df.columns:
    df = df.rename(columns = {"unyo000":"unyo009"})
    _col = sorted(df.iloc[:,1:].columns)
    df = df[["time"] + _col]
  
  if "unyo016" in df.columns:
    df = df.drop(["unyo016"],axis=1)
  
  use_col = [ c for c in df.columns if "kans" in c or "unyo" in c]
  df["mean"] = df[use_col].mean(axis=1)
  # print(df.shape)
  # sys.exit()
  return df

def load_obs(month):
  path = f"{OBS_1MIN}/{month}_1min.csv"
  df = pd.read_csv(path)
  df["time"] = pd.to_datetime(df["time"])
  df = rad_JOGAI(df,cate="obs")
  if "unyo016" in df.columns:
    df = df.drop(["unyo016"],axis=1)
  use_col = [ c for c in df.columns if "kans" in c or "unyo" in c]
  df["mean"] = df[use_col].mean(axis=1)
  return df
  

def ave1to30(df):
  for c in df.columns:
    df[c] = df[c].rolling(30).mean()
  df=df.iloc[::30,:]
  return df

def ave5to30(df):
  for c in df.columns:
    df[c] = df[c].rolling(6).mean()
  df=df.iloc[::6,:]
  return df 


def check_avetime(df):
  _index = df.index
  _hh = [ t.strftime("%M") for t in _index]
  print("unique [hhmm] -> ", np.unique(_hh))
  return

def cut_hh(df,st_hh=6,ed_hh =18):
  if "time" not in df.columns:
    df = df.reset_index()
  
  df["hh"] = df["time"].apply(lambda x: x.hour)
  df = df[(df["hh"]>=st_hh )&(df["hh"]<=ed_hh)]
  
  df = df.set_index("time")
  df = df.drop("hh",axis=1)
  return df

def cut_dd(df,dd):
  if "time" not in df.columns:
    df = df.reset_index()
  
  df["dd"] = df["time"].apply(lambda x: x.strftime("%Y%m%d"))
  df = df[df["dd"] !=dd]
  
  df = df.set_index("time")
  df = df.drop("dd",axis=1)
  return df


def plot_mm(err_name="RMSE"):
  """
  2021.07.15 最初に作成した状況
  2021.08.10 荒井さんへ報告用に修正
  2021.09.08 荒井さんへ報告用に修正/ 
  2021.09.11 荒井さんへ報告用に修正
  2021.11.05 荒井さ
  """
  # OUTD="/work/ysorimachi/hokuriku/dat2/rad/per_code"
  # ERR_D="/home/ysorimachi/work/hokuriku/dat/rad/r2"
  ERR_D="/home/ysorimachi/work/hokuriku/dat/rad/r2/per_mm/mm" #2021.11.05
  mem_col = ["kans001","kans002",'unyo001', 'unyo002', 'unyo003', 'unyo004', 'unyo005',
       'unyo006', 'unyo007', 'unyo008', 'unyo009', 'unyo010', 'unyo011',
       'unyo012', 'unyo013', 'unyo014', 'unyo015','unyo017',
       'unyo018',"mean"]
  
  renames = [ rad_code2name(c) for c in mem_col]
  # print(renames)
  # sys.exit()
  
  _month = loop_month(st="202104", ed="202510")[:6]
  _df = []
  for month in _month:
    path = f"{ERR_D}/{month}.csv"
    df = pd.read_csv(path).set_index("code")
    df.columns = [f"{c}({month})" for c in df.columns]
    _df.append(df)
  df = pd.concat(_df,axis=1)
  
  use_col = [ c for c in df.columns if err_name in c]
  df = df[use_col].T
  df.columns = renames
  # print(df.head())
  # sys.exit()
  
  f,ax = plt.subplots(figsize=(18,10))
  # ax = ax.flatten()
  
  
  for c in df.columns:
    ax.plot(df[c] ,label=c,marker="o", lw=2)
    ax.text(ax.get_xlim()[1]-0.20, df[c].values[-1], c,fontsize=8)
  
  plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=16)
  ax.set_title(err_name, fontsize=15)
  ax.set_ylabel(fr"{err_name}[W/$m^2$]")
  ax.set_xlabel("MONTH")
  # ax.set_xlim(-1,6)
  f.savefig(f"{OUT_D}/err_{err_name}.png", bbox_inches="tight")
  return


def scatter_mm(_month):
  st_mm,ed_mm = _month[0], _month[-1]

  mem_col = ["kans001","kans002",'unyo001', 'unyo002', 'unyo003', 'unyo004', 'unyo005',
       'unyo006', 'unyo007', 'unyo008', 'unyo009', 'unyo010', 'unyo011',
       'unyo012', 'unyo013', 'unyo014', 'unyo015','unyo017',
       'unyo018',"mean"]
  
  renames = [ rad_code2name(c) for c in mem_col]
  # print(renames)
  # sys.exit()
  
  f,ax = plt.subplots(4,5,figsize=(25,20))
  ax = ax.flatten()
  
  for i,c in enumerate(mem_col):
    name = rad_code2name(c)
    _df = [ pd.read_csv(f"{PER_CODE}/rad_{c}_{m}.csv") for m in _month]
    df = pd.concat(_df,axis=0)
    df["time"] = pd.to_datetime(df["time"])
    df = cut_hh(df,st_hh=6,ed_hh =18).dropna()
    
    # clensing ---
    df = df[(df["obs"]>0.1)&(df["8now0"]>0.1)]
    if c=="mean":
      df = cut_dd(df,dd="20210831")
      
    # df["r"] = df["8now0"]/df["obs"]
    # df = df.sort_values("r",ascending=False)
    # print(c,"-"*50)
    # print(df.head())
    #-------------

    ax[i].scatter(df["obs"],df["8now0"],s=1,color="r")
    ax[i].plot(np.arange(1200), np.arange(1200),lw=1, color="k")
  
  # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=16)
    ax[i].set_title(name, fontsize=15)
    ax[i].set_ylabel(r"8Now0[W/$m^2$]")
    ax[i].set_xlabel(r"OBS[W/$m^2$]")
  # ax.set_xlim(-1,6)
  
  plt.subplots_adjust(wspace=0.4, hspace=0.4)
  f.savefig(f"{OUT_D}/rad_scatter_{st_mm}to{ed_mm}.png", bbox_inches="tight")
  print(f"{OUT_D}/rad_scatter_{st_mm}to{ed_mm}.png")
  plt.close()
  return

def scatter_yy():
  # st_mm,ed_mm = _month[0], _month[-1]
  _month = loop_month(st="201904", ed="202510")[:24+6]
  # print(_month)
  # sys.exit()

  mem_col = ["kans001","kans002",'unyo001', 'unyo002', 'unyo003', 'unyo004', 'unyo005',
       'unyo006', 'unyo007', 'unyo008', 'unyo009', 'unyo010', 'unyo011',
       'unyo012', 'unyo013', 'unyo014', 'unyo015','unyo017',
       'unyo018',"mean"]
  
  renames = [ rad_code2name(c) for c in mem_col]
  # print(renames)
  # sys.exit()
  
  f,ax = plt.subplots(3,4,figsize=(4*5,3*5))
  ax = ax.flatten()
  _yy = ["2019年度","2020年度","2021年度"]
  for k,mm in enumerate(_month):
    i = int(k % 12)
    j = int(k // 12)
    title = _month[i][4:6]
    cyy = _yy[j]
    
    # name = rad_code2name(c)
    name = "mean"
    path = f"{PER_CODE}/rad_{name}_{mm}.csv"
    df = pd.read_csv(path)
    # _df = [ pd.read_csv(f"{PER_CODE}/rad_{c}_{m}.csv") for m in _month]
    # df = pd.concat(_df,axis=0)
    df["time"] = pd.to_datetime(df["time"])
    df = cut_hh(df,st_hh=6,ed_hh =18).dropna()
    
    # clensing ---
    df = df[(df["obs"]>0.1)&(df["8now0"]>0.1)]
    if name=="mean":
      df = cut_dd(df,dd="20210831")

    # print(mm)
    # print(df.head())
    # print(df.describe())
    # sys.exit()
    # df["r"] = df["8now0"]/df["obs"]
    # df = df.sort_values("r",ascending=False)
    # print(c,"-"*50)
    # print(df.head())
    #-------------

    # ax[i].scatter(df["obs"],df["8now0"],s=1,color="r")
    ax[i].scatter(df["obs"],df["8now0"],s=1,color=_color[j],label=cyy)
    ax[i].plot(np.arange(1200), np.arange(1200),lw=1, color="k")
  
  # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=16)
    ax[i].set_title(f"{title}月", fontsize=15)
    ax[i].set_ylabel(r"8Now0[W/$m^2$]")
    ax[i].set_xlabel(r"OBS[W/$m^2$]")
    ax[i].set_ylim(0,1200)
    ax[i].set_xlim(0,1200)
    ax[i].legend(loc="upper left")
    
  # ax.set_xlim(-1,6)
  
  plt.subplots_adjust(wspace=0.4, hspace=0.4)
  f.savefig(f"{OUT_D}/rad_scatter_yy.png", bbox_inches="tight")
  print(f"{OUT_D}/rad_scatter_yy.png")
  plt.close()
  return
  


if __name__ =="__main__":
  
  _month = loop_month(st="202104", ed="202510")[:6]
  # _month = loop_month(st="201904", ed="202510")[:24]
  # print(_month)
  # sys.exit()
  
  if 0:
    for err_name in ["ME","RMSE"]:
      plot_mm(err_name) #2021.08.10/09.08(update) -> 2019/2020 code別のscatter
  
  if 1:
    # scatter_mm(_month)
    scatter_yy()