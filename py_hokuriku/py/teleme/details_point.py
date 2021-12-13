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
from utils_teleme import loop_month, sonzai_sort
from mapbox import map_lonlat3 #(df,html_path,zoom=4,size_max=None)

TELEME="/work/ysorimachi/hokuriku/dat2/teleme/dataset" #30min_201912.csv

  
def check_null():
  
  def null_teleme(month="202004"):
    path = f"{TELEME}/30min_{month}.csv"
    df = pd.read_csv(path)
    null = (1- df.isnull().sum()/df.shape[0]).sort_values(ascending=False).drop("time").reset_index(drop=True)
    # null.to_csv(f"/home/ysorimachi/work/hokuriku/dat/teleme/null/count/{month}_null.csv")
    return null
  
  _mm = loop_month(st="201904")
  _null= []
  for mm in _mm:
    n = null_teleme(month=mm)
    _null.append(n)
  
  df = pd.concat(_null,axis=1)
  df.columns = _mm
  f,ax = plt.subplots(figsize = (18,7))
  
  for c in df.columns:
    ax.plot(df[c], label=c)
  ax.set_ylim(0,1)
  ax.legend()
  ax.set_ylabel("ratio[isData]")
  ax.set_xlabel("count")
  f.savefig(f"/home/ysorimachi/work/hokuriku/dat/teleme/null/count/null.png",bbox_inches="tight")

def plot_map_teleme():
  """
  2021.07.16作成
  telemeと日射量観測地点の分布を表示する
  """
  df, _ = mk_teleme_table()
  rad = mk_rad_table()
  use_col = ["lon","lat","text"]
  df["text"] = df[["name","code","max","panel"]].apply(lambda x:f"{x[0]}-{x[1]}[{x[2]}({x[3]})]",axis=1)
  df = df.dropna(subset = ["lon","lat"])
  df = df[use_col]
  df["size"] = 2
  rad["code"] = rad["code"].apply(lambda x: np.nan if x.startswith("kepv") else x)
  rad = rad.dropna(subset=["code"])
  rad["text"] = rad[["code","name"]].apply(lambda x:f"{x[0]}-{x[1]}",axis=1)
  rad=rad[use_col]
  rad["size"] = 8
  
  df = pd.concat([df,rad],axis=0)
  html_path = "/home/ysorimachi/work/hokuriku/dat/teleme/map/html/map.html"
  map_lonlat3(df,html_path,zoom=4,size_max=None)
  print(html_path)
  return


def hist_teleme(cate="max",plot_type="count"):
  df = detail_teleme(CSV=True)
  tl = mk_teleme_table(ONLY_df=True)
  OUTD="/home/ysorimachi/work/hokuriku/out/teleme/detail/hist"
  #-----------------------------------
  def select_teleme(_index):
    tmp = tl.loc[tl["code"].isin(_index)]
    return tmp
  #-----------------------------------
  
  _mm  = df.columns
  _mm=["201904","202004","202103"]
  # print(_mm)
  # sys.exit()
  f,ax = plt.subplots(figsize=(12,5))
  
  _tmp=[]
  for mm in _mm:
    n_all = df.loc[df[mm]>0.0,mm].shape[0]
    _index = df.loc[df[mm]>0.6,mm].index
    n_select = len(_index)
    tmp = select_teleme(_index)
    # print(tmp.head())
    tmp["month"] = mm
    _tmp.append(tmp)
  df = pd.concat(_tmp,axis=0)
  if plot_type =="count":
    sns.histplot(data=df,x=cate,ax=ax,hue="month",bins=30,element="step",kde=True)
  if plot_type == "density":
    # sns.displot(data=penguins, x="flipper_length_mm", hue="species", kind="kde")
    # print(df.head())
    # sys.exit()
    sns.kdeplot(data=df,x=cate,ax=ax,hue="month")
  f.savefig(f"{OUTD}/hist_{plot_type}_teleme_{cate}.png", bbox_inches="tight")
  return


def detail_teleme(CSV=False):
  ""
  TELEME="/work/ysorimachi/hokuriku/dat2/teleme/dataset"
  _month = loop_month()
  _res = []
  
  
  for month in _month:
    path = f"{TELEME}/30min_{month}.csv"
    df = pd.read_csv(path)
    
    use_col = [ c for c in df.columns if "telm" in c ]
    n = df.shape[0]
    res = (n - df.isnull().sum())/n
    res = res[use_col]
    res.name = month
    _res.append(res)
  
  df = pd.concat(_res,axis=1)
  df = df.round(2)
  
  if 0:
    f,ax = plt.subplots(figsize=(15,20))
    sns.heatmap(df,annot=True,vmin=0,vmax=1,ax=ax)
    ax.set_title("テレメデータ存在率[0=欠測]")
    f.savefig("/home/ysorimachi/work/hokuriku/dat/teleme/null/count/heatmap.png", bbox_inches="tight")
  return df
  



if __name__ == "__main__":
  
  if 1:
    """
    2021.07.16 add 
    初期解析に必要な描画や欠測(null)についての表示
    """
    # plot_map_teleme()
    # sys.exit()
    # check_null() #kessoku
    df = detail_teleme(CSV=True) #heatmap(2021.010.01)
    print(df.head())
    sys.exit()
  
  
  if 0:
    """
    2021.07.21 add 
    初期解析に必要な描画や欠測(null)についての表示
    """
    cate="max"
    # plot_type="count"
    for plot_type in ["density","count"]:
      hist_teleme(cate,plot_type)
      
  if 1:
    sonzai_sort()