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
from utils import *

SET="/work/ysorimachi/hokuriku/dat2/load/set1"
def load_load(month="202004", ave=False):
  path = f"{SET}/{month}.csv"
  df = pd.read_csv(path)
  ind_col = [ c for c in df.columns if f"H9" in c]
  
  df = clensing_col(df,_col=ind_col)
  if ave:
    
    for cate in [ "IN","EX","PV"]:
      use_col = [ c for c in df.columns if f"_{cate}" in c]
      
      df[f"min_{cate}"] = df[use_col].min(axis=1)
      df[f"max_{cate}"] = df[use_col].max(axis=1)
      df[f"mean_{cate}"] = df[use_col].mean(axis=1)
      df[f"std_{cate}"] = df[use_col].std(axis=1)
      df[f"sum_{cate}"] = df[use_col].sum(axis=1)
      
    #自家消費量の計算
    df[f"sum_USE"] = df[f"sum_PV"] -df["sum_EX"]
    df = df.drop(ind_col, axis=1)
    return df
  else:
    return df
    

def load_load0(month="201905"):
  
  # path = f"{SET}/{month}.csv"
  # df = pd.read_csv(path)
  df = load_load(month=month, ave=True)
  print(df.head())


if __name__ == "__main__":
  load_load0()
  
  