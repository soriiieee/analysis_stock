# -*- coding: utf-8 -*-
# when   : 2020.0x.xx
# who : [sori-machi]
# what : [ ]
#---------------------------------------------------------------------------
# basic-module
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
_color = [ mcolors.rgb2hex(plt.cm.get_cmap('tab10')(i)) for i in range(10)]

import sys,os,re,glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
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

from utils import *
# os.makedirs(outd, exist_ok=True)
# subprocess.run('rm -f *.png', cwd=outd, shell=True)


def time_Series(_nation, cate="confirmed"):
  cpath = f"{TS_DATA}/time_series_covid19_{cate}_global.csv"
  df = pd.read_csv(cpath)
  df = df.loc[df["Country/Region"].isin(_nation),:]
  df = df.drop(["Province/State","Lat","Long"],axis=1)
  df = df.groupby("Country/Region").sum().T
  df = df[_nation]
  df.index.name = "time"
  df = df.reset_index()
  df["time"] = df["time"].apply(lambda x: conv_time(x))
  df = df.set_index("time")
  df.to_csv(f"{OUT}/ts_{cate}.csv")
  return 

def plt_ts(_nation):
  
  def load_data(nation):
    _df = []
    for cate in ["confirmed","deaths","recovered"]:
      path = f"{OUT}/ts_{cate}.csv"
      df = pd.read_csv(path)
      df["time"] = pd.to_datetime(df["time"])
      df = df.set_index("time")
      df = df[nation]
      col = f"{nation}-[{cate}]"
      df.name = col
      _df.append(df)
    df = pd.concat(_df,axis=1)
    return df
  
  def plot_ts(ax,df,title):
    vmax = 1.1*(df.describe().T["max"].max())
    
    for i,c in enumerate(df.columns):
      color = _color[i]
      # ax.plot(np.arange(len(df)), df[c],label=c)
      ax.plot(df[c],label=c, color=color)
    
    # --------------------------------------------
    # ratio ---(増加率) ---------------
    # bx = ax.twinx()
    # bx.set_ylim(-0.3,0.3)
    # for i,c in enumerate(df.columns):
    #   color = _color[i]
    #   if i==0 or i==1:
    #     df[c] = df[c].pct_change()
    #     bx.plot(df[c], label=c, lw=5,color=color)
    # bx.axhline(y=0,color="k", lw=1)
    # ratio ---(増加率) ---------------
    # --------------------------------------------
    
    ax.legend(loc="upper left")
    ax.set_ylim(0,vmax)
    ax.set_title(title, fontsize=18)
    return ax
  
  def day_diff(df,MA=7):
    for c in df.columns:
      df[c] = df[c].diff()
    df = df.dropna()
    if MA:
      for c in df.columns:
        df[c] = df[c].rolling(MA).mean()
    return df
  
  def day_pct(df,MA=7):
    for c in df.columns:
      df[c] = df[c].rolling(MA).mean()
      df[c] = df[c].pct_change()
    df = df.dropna()
    return df
  
  for i,nation in enumerate(_nation):
    ci = str(i+1).zfill(5)
    df = load_data(nation) # 累計人数(人)
    
    # df = cut_time(df, st=datetime(2021,4,1,0,0), ed=None)
    df = day_diff(df,MA=7)
    # df = day_pct(df,MA=7)
    f,ax = plt.subplots(figsize=(15,8))
    ax = plot_ts(ax,df,title=f"{nation}-> [ comsum ]")
    f.savefig(f"{OUT}/nation_ts/{ci}_{nation}.png", bbox_inches="tight")
    print(datetime.now(),"[END]",ci,nation)
    # sys.exit()

def cleaner_png(DIR):
  subprocess.run(f"rm {DIR}/*.png", shell=True)
  return


if __name__ == "__main__":
  dd= "20211130"
  N=100
  # _nation = select_dd(dd=dd, N=N) #2021.11.03
  _nation = select_dd2(dd=dd, N=N) #2021.12.02
  # cleaner_png(DIR="../out/nation_ts")
  print(_nation)
  sys.exit()
  
  if 1:
    """
    making dataset ----(2021.10.26)
    making dataset ----(2021.11.05)
    """
    for cate in ["confirmed","deaths","recovered"]:
      time_Series(_nation,cate)
  
  if 0:
    # _nation = _nation[:30]
    df = select_dd(dd="202010", N=30, cate="Confirmed", return_df=True)
    df = select_dd(dd=dd, N=30, cate="Confirmed", return_df=True)
    df2 = check_dd2(dd)
    df = pd.concat([df2,df],axis=1)
    print(df.head())
    sys.exit()
