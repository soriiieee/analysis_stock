# -*- coding: utf-8 -*-
# when   : 2020.0x.xx
# who : [sori-machi]
# what : [ ]
#---------------------------------------------------------------------------
# basic-module
import matplotlib
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
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
#amedas relate 2020.02,04 making...
from tool_AMeDaS import code2name, name2code
from tool_110570 import get_110570,open_110570
from tool_100571 import get_100571,open_100571,get_100571_val
#(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
import subprocess
# outd ='/home/griduser/work/sori-py2/deep/out/0924_01'
# os.makedirs(outd, exist_ok=True)
# subprocess.run('rm -f *.png', cwd=outd, shell=True)
from utils_snow import load_snowdepth_dd

def main(yy):
  _name=["氷見","富山","魚津","珠洲","輪島","金沢","白山河内","福井","敦賀","加賀菅谷"]
  # _name = ["氷見","富山","朝日","魚津","珠洲","輪島","七尾","金沢","白山河内","加賀菅谷","大野","福井","敦賀","小浜"]
  in_d="/home/ysorimachi/work/hokuriku/out/snow2/csv"
  _file = sorted(glob.glob(f"{in_d}/111095_{yy}*.csv"))
  # print(len(_file))
  # sys.exit()
  # _file = sorted(glob.glob(f"{in_d}/111095_*.csv"))
  _ini_j = [ os.path.basename(path).split(".")[0].split("_")[1] for path in _file]
  _file2 = [ f"{in_d}/121211_{ini_j}.csv" for ini_j in _ini_j]
  for name in _name:
    code = name2code(name)
    _val0,_val1,_val_ame= [],[],[]
    for path0,path1,ini_j in zip(_file,_file2, _ini_j):
      d0 = pd.read_csv(path0, encoding="shift-jis")
      d1 = pd.read_csv(path1, encoding="shift-jis")

      _val0.append(d0.loc[d0["name"]==name,"val"].values[0])
      _val1.append(d1.loc[d1["name"]==name,"val"].values[0])
      
      depth = get_100571_val(code, ini_j, val="snowDepth")
      _val_ame.append(depth)

    
    df = pd.DataFrame()
    df["time"] = _ini_j
    df["111095"] = _val0
    df["121211"] = _val1
    df["AMeDaS"] = _val_ame
    df.to_csv(f"/home/ysorimachi/work/hokuriku/out/snow2/csv_ts/{name}_{yy}.csv", index=False)
    print("end", name)
  return

def plot():
  in_d="/home/ysorimachi/work/hokuriku/out/snow2/csv_ts"
  _name=["氷見","富山","魚津","珠洲","輪島","金沢","白山河内","福井","敦賀","加賀菅谷"]

  for name in _name:
    df = pd.read_csv(f"{in_d}/{name}.csv")
    df["time"] = pd.to_datetime(df["time"].astype(str))

    # -----------------------------------------
    f,ax = plt.subplots(figsize=(15,6))
    for c in ["111095","AMeDaS"]:
      ax.plot(df["time"].values,df[c].values, label=c)
    ax.legend(loc="upper left")
    ax.set_xlabel("time[day(JST)]")
    ax.set_ylabel("depth[cm]")
    ax.set_ylim(0,20)
    ax.set_title(name)
    f.savefig(f"/home/ysorimachi/work/hokuriku/out/snow2/csv_ts/png/{name}.png", bbox_inches="tight")
    # -----------------------------------------
  return

def plot2():
  name="富山"
  d1= pd.read_csv(f"/home/ysorimachi/work/hokuriku/out/snow2/csv_ts/{name}_2020.csv")
  d1 = d1.rename(columns={"AMeDaS":"2020"})
  d2= pd.read_csv(f"/home/ysorimachi/work/hokuriku/out/snow2/csv_ts/{name}_2021.csv")
  d2 = d2.rename(columns={"AMeDaS":"2021"})

  for df in [d1,d2]:
    df["time"] = df["time"].astype(str).apply(lambda x: x[4:])
  
  d1 = d1.merge(d2, on="time", how="inner")
  d1["time"] = d1["time"].apply(lambda x: "2020"+x)
  d1["time"] = pd.to_datetime(d1["time"])

  # -----------------------------------------
  f,ax = plt.subplots(figsize=(15,6))
  for c in ["2020","2021"]:
    ax.plot(d1["time"].values,d1[c].values, label=c)
  ax.legend(loc="upper left")
  ax.set_xlabel("time[day(JST)]")
  ax.set_ylabel("depth[cm]")
  ax.set_ylim(0,40)
  ax.set_title(name)
  f.savefig(f"/home/ysorimachi/work/hokuriku/out/snow2/csv_ts/png/{name}_2y.png", bbox_inches="tight")
  # -----------------------------------------

def ts_plot(dd="20210110"):
  """[summary]
  init: 2021.12.10 add 
  all teleme points plotting ...
  Args:
      dd (str, optional): [description]. Defaults to "20210110".
  """
  OUT_D="/home/ysorimachi/work/hokuriku/out/snow/png"
  df = load_snowdepth_dd(dd)
  # print(df.head())
  n_all = len(list(df.describe().T["max"].dropna()))
  tmp = df.describe().T["max"].dropna().reset_index()
  thres_v = 20
  tmp = tmp[tmp["max"]>thres_v]
  print(tmp.shape[0], n_all, tmp.shape[0]/n_all)
  # sys.exit()
  
  f,ax = plt.subplots(6,10,figsize=(50,25))
  ax = ax.flatten()
  
  use_col = list(df.describe().T["max"].dropna().index)
  # use_col = [ c for c in df.columns if "telm" in c][:60]
  
  # print(len(use_col))
  # sys.exit()
  rcParams['font.size'] = 18
  for i,c in enumerate(use_col):
    alpha=0.4
    color="blue"
    ax[i].bar(np.arange(len(df)),df[c],color=color, alpha=alpha)
    # pv_max = teleme_max(c)
    title = f"{c}"
    ax[i].set_title(title)
    ax[i].tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
    ax[i].set_ylim(0,200)
    ax[i].axhline(y=thres_v, color="r", lw=1)
  
    if df[c].values[0] ==np.nan:
      ax[i].set_visible(False)
  # for i in range(df.shape[1],60):
  #   ax[i].set_visible(False)
    
  # ax.set_title(f"双方向端末データ-PV発電量(実況-予測) -{dd}")
  # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
  # ax.set_ylabel(ylabel)
  
  #----------xaxis ---------
  # ax.set_xlim(0,len(df)) #x軸の限界値を設定
  # st, ed = ax.get_xlim()
  # ax.xaxis.set_ticks(np.arange(int(st), int(ed),dt))
  # ax.set_xticklabels(_t[::dt])
  
  #----------xaxis ---------
  plt.subplots_adjust(wspace=0.2, hspace=0.2)
  f.savefig(f"{OUT_D}/point_{dd}.png", bbox_inches="tight")
  plt.close()
  print(datetime.now(),"[END]", f"{OUT_D}")
  return

if __name__=="__main__":
  yy=2021
  # main(yy)
  # plot2()
  ts_plot(dd="20210110")