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
# from getErrorValues import me,rmse,mae,r2 #(x,y)
# from convSokuhouData import conv_sfc #(df, ave=minutes,hour)
# #amedas relate 2020.02,04 making...
# from tool_AMeDaS import code2name, name2code
# from tool_110570 import get_110570,open_110570
# from tool_100571 import get_100571,open_100571
# #(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
import subprocess
# outd ='/home/griduser/work/sori-py2/deep/out/0924_01'
# os.makedirs(outd, exist_ok=True)
# subprocess.run('rm -f *.png', cwd=outd, shell=True)


def plot_ts(df,c,png_path="./plt_show.png"):
  f,ax = plt.subplots(figsize=(18,8))
  if type(c)==list:
    for col in c:
      ax.plot(df[col], label=col)
  else:
    ax.plot(df[c], label=c)
  f.savefig(png_path, bbox_inches="tight")
  return

def plot_hist(df,c,png_path="./plt_show.png"):
  f,ax = plt.subplots(figsize=(18,8))
  # ax.plot(df[c], label=c)
  sns.distplot(df[c],ax=ax)
  f.savefig(png_path, bbox_inches="tight")
  return