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
EC_HOME = "/work2/ysorimachi/ec_maeda_210709"
MIX_HOME ="/work2/ysorimachi/synfos/data/SYNFOS-solar_ver2/maeda_210709/store"
OHOME="/home/ysorimachi/work/synfos/tmp/maeda0709/null"


def check_log():
  """
  2021.07.14
  ログファイルをチェックして、表示する/プログラム
  """
  path = "./data.log"
  df = pd.read_csv(path, delim_whitespace=True,names=["time","status","code","ini_u"])
  print("All num => ",df.shape)
  tmp =df[df["status"]=="[error]"]
  print("[error] num => ",tmp.shape)
  print("end code Num => ",df["code"].nunique())
  tmp.to_csv(f"{OHOME}/tmp_null.csv")
  return

def check_ec_syn():
  code="hkdn001"
  # mtd=1
  path = "./data.log"

  df = pd.read_csv(path, delim_whitespace=True,names=["time","status","code","ini_u"])
  _ini_u = sorted(df["ini_u"].unique().tolist())
  print(len(_ini_u))
  _res=[]
  # _ini_u2 = _ini_u[:10]
  _ini_u2 = _ini_u
  _ini_u2.remove(201903192100)
  
  for ini_u in tqdm(_ini_u2):
    # try:
    path = f"{MIX_HOME}/{ini_u}/{code}.csv"
    df = pd.read_csv(path)
    df = df.drop(0).replace(9999,np.nan)
    res = df.isnull().sum()
    _res.append(pd.Series(res))
  
  df = pd.concat(_res,axis=1).T
  df.index = _ini_u2
  df.index.name = "ini_U"
  df.to_csv(f"{OHOME}/ec_syn_null.csv")
  return

def null_analysis():
  path = f"{OHOME}/ec_syn_null.csv"
  df = pd.read_csv(path)
  syn = df[df["SYN"] !=0]
  ec = df[df["ECs0"] !=0]
  print(syn.head())
  print(ec.head())
  
  syn.to_csv(f"{OHOME}/syn_null.csv",index=False)
  ec.to_csv(f"{OHOME}/ec_null.csv",index=False)
  print(syn.shape[0],ec.shape[0])
  sys.exit()
  return
    
if __name__ == "__main__":
  
  if 0:
    # check_log()
    check_ec_syn()
  if 1:
    null_analysis()



# print(df.tail())
# print(tmp)
