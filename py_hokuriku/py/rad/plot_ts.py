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
from getErrorValues import me,rmse,mae,r2 #(x,y)
from convSokuhouData import conv_sfc #(df, ave=minutes,hour)
#amedas relate 2020.02,04 making...
# from tool_AMeDaS import code2name, name2code
# from tool_110570 import get_110570,open_110570
# from tool_100571 import get_100571,open_100571
#(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
import subprocess
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
# rcParams['font.size'] = 18
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
import matplotlib.colors as mcolors
_color = [ mcolors.rgb2hex(plt.cm.get_cmap('tab20')(i)) for i in range(20)]

sys.path.append("..")
# from utils import *
# def outlier_day():
#   _dd = ["20200518","20201115","20201215","20201216","20210224","20210225"]
#   return _dd

DIR_1MIN="/work/ysorimachi/hokuriku/dat2/rad/obs/1min"
def load_rad1min(dd):
  month = str(dd)[:6]
  path = f"{DIR_1MIN}/{month}_1min.csv"
  df = pd.read_csv(path)
  df["time"] = pd.to_datetime(df["time"])
  df["day"] = df["time"].apply(lambda x: x.strftime("%Y%m%d"))
  df =df[df["day"]==dd]
  df = df.drop(["kans001","kans002","day"], axis=1)
  df = df.set_index("time")
  return df

def rad_code2name(code="unyo001"):
  if code == "mean" or code == "ave":
    return "エリア平均"
  path = f"/home/ysorimachi/work/hokuriku/dat/rad/re_get/list_rad_point2.csv"
  df = pd.read_csv(path).set_index("code")
  name_dict = df["name"].to_dict()
  # print(name_dict)
  
  if code=="kans001":
    return "富山気象官署"
  elif code=="kans002":
    return "福井気象官署"
  else:  
    return name_dict[code]

def  plot_rad_day(dd,code=None):
  # from matplotlib import rcParams
  TS_OUT_DD="/home/ysorimachi/work/hokuriku/dat/rad/ts/obs_dd"
  df = load_rad1min(dd)

  df["mean"] = df.mean(axis=1)
  
  # print(df.head())
  # sys.exit()
  # re_col = [ rad_code2name(c) for c in df.columns ]
  
  #plot figure
  rcParams['font.size'] = 36
  rcParams['ytick.labelsize'] = 24
  rcParams['xtick.labelsize'] = 24
  
  vmin,vmax=0,1200
  f,ax = plt.subplots(figsize=(18,8))
  png_path = f"{TS_OUT_DD}/unyo_{dd}.png"
  
  for i,c in enumerate(df.columns):
    if c != "mean":
      color = _color[i]
      lw, alpha = 1,1
    else:
      color = "k"
      lw, alpha = 5,1
    # if code:
    #   if c==code:
    #     lw, alpha = 8,1
    #     color = "b"
    #   else:
    #     lw, alpha = 2,0.5
    #     color = "gray"
    # else:
    #   lw, alpha = 1,1
    #   color = "k"
    
    name = rad_code2name(c)
    label = f"{c}({name})"
    ax.plot(np.arange(len(df)), df[c].values, label=label,lw=lw,alpha=alpha, color=color)
  
  # ax.set_xlabel("時刻[HH:MM]")
  ax.set_ylabel("全天日射量[W/m2]")
  ax.set_ylim(vmin,vmax)
  ax.set_xticks(np.arange(len(df)))
  
  step=60*3
  _t = [ t.strftime("%H:%M") for t in df.index]
  ax.set_xticks(np.arange(len(df)))
  ax.set_xlim(0,len(df))
  st, ed = ax.get_xlim()
  ax.xaxis.set_ticks(np.arange(int(st), int(ed),step))
  ax.set_xticklabels(_t[::step])
  ax.legend(loc="upper right",fontsize=12)
  
  title = f"{dd}(運用観測点の日射量)"
  ax.set_title(title)
  f.savefig(png_path, bbox_inches="tight")
  plt.close()
  print(datetime.now(), "[END]", dd, code)
  print(png_path)
  
  return

def loop_dd():
  _dd = ["20200518","20201115","20201215","20201216","20210224","20210225"]
  _code = ["unyo018","unyo012","unyo005","unyo005","unyo002","unyo002"]
  return _dd,_code

def main():
  _dd,_code = loop_dd()
  
  for code,dd in zip(_code,_dd):
    plot_rad_day(dd,code=code)
    # sys.exit()

def main2(dd="20200614"):
  plot_rad_day(dd)
  return 
  

if __name__ == "__main__":
  # main()
  dd="20200614"
  
  main2(dd) #2021.12.10