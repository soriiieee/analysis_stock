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
from plot1m import plot1m, plot1m_ec#(df,_col,_mem_col=False,month=False,vmin=0,vmax=1000,title=False)
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
from utils_load import *

SET="/work/ysorimachi/hokuriku/dat2/load/set1"
TS_OUT="/home/ysorimachi/work/hokuriku/out/load/ts"
CHECK_OUT="/home/ysorimachi/work/hokuriku/out/load/check"

_month = loop_month()
def check():
  for month in _month:
    df = load_load(month=month, ave=True)
    v_col = [ c for c in df.columns if "sum" in c]
    use_col = ["time"]+v_col
    
    for c in v_col:
      df[c] = df[c]/1000
      
    df = df[use_col]
    vmax = np.max(df.describe().T["max"])
    # sys.exit()
    # continue
    
    f = plot1m_ec(df,_col=[ c for c in df.columns if "sum" in c],_mem_col=False,month=month,vmin=0,vmax=vmax,title=month)
    f.savefig(f"{TS_OUT}/{month}.png",bbox_inches="tight")
    plt.close()
    print(datetime.now(),"[end]", month)
    # sys.exit()
  return

def null_count():
  _count=[]
  for month in _month:
    df = load_load(month=month, ave=False)
    pv_col = ["time"] + [c for c in df.columns if "_PV" in c]
    df = df[pv_col]
    n_data = df.shape[0]
    res = (1- df.isnull().sum()/n_data).sort_values(ascending=False)
    
    n = res[res>0.9].shape[0]
    _count.append(n)
  
  df =pd.DataFrame()
  df["month"] = _month
  df["count"] = _count
  df.to_csv(f"{CHECK_OUT}/null_count.csv")


if __name__ == "__main__":
  # check() #月ごとの合計発電量/自家消費電力量のplot
  null_count()
  