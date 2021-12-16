# -*- coding: utf-8 -*-
# when   : 2020.03.23 
# who : [sori-machi]
# what : [ ]
"""
作成したsynfosがしっかりとデータとして表示されているのかを確認するファイル
"""
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
# from getErrorValues import me,rmse,mae,r2 #(x,y)
# from convSokuhouData import conv_sfc #(df, ave=minutes,hour)
# #amedas relate 2020.02,04 making...
# from tool_AMeDaS import code2name, name2code
# from tool_110570 import get_110570,open_110570
# from tool_100571 import get_100571,open_100571
#(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
import subprocess
# outd ='/home/griduser/work/sori-py2/deep/out/0924_01'
# os.makedirs(outd, exist_ok=True)
# subprocess.run('rm -f *.png', cwd=outd, shell=True)

def list_dir(N=None):
  DHOME="/mnt/ysorimachi/synfos/mix"
  _list = os.listdir(DHOME)
  LIST=sorted([ f"{DHOME}/{l}"for l in _list])
  if N is not None:
    LIST = LIST[:N]
  return LIST

def clensing_ec(df):
  df = df.replace(9999,np.nan)
  df.loc[df["flg_ecm"]==1,"iecm"] = np.nan
  df.loc[df["flg_ecc"]==1,"iecc"] = np.nan
  return df

def load_data(path):
  # names=["time","rrad","isyn","iecm","iecc","flg_ecm","flg_ecc"]
  names=["time","rrad","isyn","iecm","iecc","flg_ecm","flg_ecc","rCR0","rS0"]
  df = pd.read_csv(path, delim_whitespace=True,header=None,names=names)
  # df = pd.read_csv(path
  
  df["time"] = pd.to_datetime(df["time"].astype(str))
  df = clensing_ec(df)
  return df

def plot_line(ax,df,title="plot"):
  # for col in ["rrad","isyn","iecm"]:
  for col in ["rrad","isyn","iecm","rCR0","rS0"]:
    if col =="rrad":
      ax.plot(df[col].values, label=col, lw=5)
    else:   
      ax.plot(df[col].values, label=col)
  ax.set_ylim(0,1000)
  ax.set_ylabel("rad[W/m2]")
  ax.set_xlabel("Forecast Time[jt]")
  ax.set_title(title)
  ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
  return ax

def main():
  """
  main function 
  init 21.03.23
  update 21.06.26
  """
  MTD=2
  cpnt = "hkrk001"
  DIR = f"/work2/ysorimachi/synfos/data/SYNFOS-solar_ver2/okada_hokuriku_0624/{MTD}"
  _dd = sorted(os.listdir(DIR))[:8]
  
  f,ax = plt.subplots(int(len(_dd)/2),2,figsize=(8,8))
  ax = ax.flatten()
  
  # for i,zz in enumerate(_zz):
  for j,dd in enumerate(_dd):
    path = f"{DIR}/{dd}/{cpnt}.dat"
    
    df = load_data(path)
    
    title = f"[{cpnt}]({dd})JST"
    ax[j] = plot_line(ax[j],df,title=title)
      # print(df.head())
  # plt.subplots_adjust(wspace=0.4, hspace=0.5)
  plt.subplots_adjust(hspace=0.8,wspace=0.7)
  plt.savefig(f"/home/ysorimachi/work/synfos/tmp/png/{cpnt}_MTD{MTD}.png",bbox_inches="tight")
  print("/home/ysorimachi/work/synfos/tmp/png")
  sys.exit()
  return


if __name__ == "__main__":
  main()
  