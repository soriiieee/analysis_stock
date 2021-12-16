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

sys.path.append("/home/ysorimachi/work/synfos/py/som_data")
from a01_99_utils import *
from c01_som_cluster import *
from x99_pre_dataset import load_rad, load_10


def load_cate(n=1, train="ALL"):
  path = glob.glob(f"/home/ysorimachi/data/synfos/cate/DAY_cate{n}*.csv")[0]
  df = pd.read_csv(path)
  df = df.sort_values("dd").reset_index(drop=True)
  
  df["time"] = df["dd"].apply(lambda x: pd.to_datetime(f"{x}0000"))
  df = train_flg(df)
  df["dd"] = df["time"].apply(lambda x: x.strftime("%Y%m%d"))
  df = df.drop(["time"],axis=1)
  
  if n==1:
    cname = "DAY_MAX_CATE"
  elif n==2:
    cname = "DAY_CI"
  
  if train == "ALL":
    _dd = df["dd"].values
    _ll = df[cname].values
    return _dd,_ll
  else:
    if train:
      df = df[df["istrain"]==1]
    else:
      df = df[df["istrain"]==0]
    _dd = df["dd"].values
    _ll = df[cname].values
    return _dd,_ll


def main():
  _dd,_ll = load_cate(1,train="ALL")

if __name__=="__main__":
  main()