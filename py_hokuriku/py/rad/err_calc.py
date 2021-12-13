# -*- coding: utf-8 -*-
# when   : 2021.07.15
# who : [sori-machi]
# what : 北陸技研の記録についてセットアップする(異常値等の確認)
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
from getErrorValues import me,rmse,mape,r2 #(x,y)
from convSokuhouData import conv_sfc #(df, ave=minutes,hour)
#amedas relate 2020.02,04 making...
from tool_AMeDaS import code2name, name2code
from tool_110570 import get_110570,open_110570
from tool_100571 import get_100571,open_100571
#(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
import subprocess

from plot1m import plot1m, plot1m_ec, plot1m_2axis #(df,_col,_mem_col=False,month=False,vmin=0,vmax=1000,title=False)
from plot1d import plot1d_ec #title=False)
sys.path.append("..")
from utils import *
from sklearn.linear_model import LinearRegression

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

# outd ='/home/griduser/work/sori-py2/deep/out/0924_01'
# os.makedirs(outd, exist_ok=True)
# subprocess.run('rm -f *.png', cwd=outd, shell=True)

from clensing import rad_JOGAI #(df,cate="obs")

RAD_DAT="/home/ysorimachi/work/hokuriku/dat/rad/re_get"
OBS_1MIN = "/work/ysorimachi/hokuriku/dat2/rad/obs/1min"
NOW_5MIN = "/work/ysorimachi/hokuriku/dat2/rad/8Now0/5min"

ERR="/home/ysorimachi/work/hokuriku/out/rad/err"

PER_CODE="/work/ysorimachi/hokuriku/dat2/rad/per_code"

# def clensing_rad(df):
#   df["time"] = pd.to_datetime(df["time"])
  

def load_8now0(month):
  path = f"{NOW_5MIN}/{month}_sat.csv"
  df = pd.read_csv(path)
  df["time"] = pd.to_datetime(df["time"])
  df = rad_JOGAI(df,cate="8now0")
  
  if "unyo000" in df.columns:
    df = df.rename(columns = {"unyo000":"unyo009"})
    _col = sorted(df.iloc[:,1:].columns)
    df = df[["time"] + _col]
  
  if "unyo016" in df.columns:
    df = df.drop(["unyo016"],axis=1)
  
  use_col = [ c for c in df.columns if "kans" in c or "unyo" in c]
  df["mean"] = df[use_col].mean(axis=1)
  # print(df.shape)
  # sys.exit()
  return df

def load_obs(month):
  path = f"{OBS_1MIN}/{month}_1min.csv"
  df = pd.read_csv(path)
  df["time"] = pd.to_datetime(df["time"])
  df = rad_JOGAI(df,cate="obs")
  if "unyo016" in df.columns:
    df = df.drop(["unyo016"],axis=1)
  use_col = [ c for c in df.columns if "kans" in c or "unyo" in c]
  df["mean"] = df[use_col].mean(axis=1)
  return df
  

def ave1to30(df):
  for c in df.columns:
    df[c] = df[c].rolling(30).mean()
  df=df.iloc[::30,:]
  return df

def ave5to30(df):
  for c in df.columns:
    df[c] = df[c].rolling(6).mean()
  df=df.iloc[::6,:]
  return df 


def check_avetime(df):
  _index = df.index
  _hh = [ t.strftime("%M") for t in _index]
  print("unique [hhmm] -> ", np.unique(_hh))
  return

def cut_hh(df,st_hh=6,ed_hh =18):
  if "time" not in df.columns:
    df = df.reset_index()
  
  df["hh"] = df["time"].apply(lambda x: x.hour)
  df = df[(df["hh"]>=st_hh )&(df["hh"]<=ed_hh)]
  
  df = df.set_index("time")
  df = df.drop("hh",axis=1)
  return df
  
  
def r2_mm(month,isConcatSave=False):
  """
  2021.07.15 最初に作成した状況
  2021.08.10 荒井さんへ報告用に修正
  2021.09.08 荒井さんへ報告用に修正/ 
  2021.09.11 荒井さんへ報告用に修正
  2021.11.05 荒井さ
  """
  # OUTD="/work/ysorimachi/hokuriku/dat2/rad/per_code"
  # ERR_D="/home/ysorimachi/work/hokuriku/dat/rad/r2"
  ERR_D="/home/ysorimachi/work/hokuriku/dat/rad/r2/per_mm/mm" #2021.11.05
  mem_col = ["kans001","kans002",'unyo001', 'unyo002', 'unyo003', 'unyo004', 'unyo005',
       'unyo006', 'unyo007', 'unyo008', 'unyo009', 'unyo010', 'unyo011',
       'unyo012', 'unyo013', 'unyo014', 'unyo015','unyo017',
       'unyo018',"mean"]
  
  # month = "202103"
  rad1 = load_obs(month).set_index("time")
  rad1 = ave1to30(rad1)
  rad2 = load_8now0(month).set_index("time")
  rad2 = ave5to30(rad2)
  
  # check_avetime(rad1) #debug
  # check_avetime(rad2) #debug
  # sys.exit()
  _err = {}
  for c in mem_col:
    df = pd.concat([rad1[c],rad2[c]] , axis=1)
    df.columns = ["obs","8now0"]
    # -------------2種類のradを保存するかどうか?--------
    if isConcatSave:
      df.to_csv(f"{PER_CODE}/rad_{c}_{month}.csv")
    # -------------2種類のradを保存するかどうか?--------
    df = cut_hh(df,st_hh=6,ed_hh=18)
    # print(df.head())
    # sys.exit()
    #----
    e1 = me(df["obs"],df["8now0"])
    e2 = rmse(df["obs"],df["8now0"])
    e4 = r2(df["obs"],df["8now0"])
    df = cut_hh(df,st_hh=9,ed_hh=15)
    e3 = mape(df["obs"],df["8now0"])
    _err[c] = [e1,e2,e3,e4]
  
  df = pd.DataFrame(_err).T
  df.index.name = "code"
  df.columns = ["ME","RMSE","MAPE","r2"]
  
  df.to_csv(f"{ERR_D}/{month}.csv")
  print(datetime.now(),"[END]", month)
  return
    
  


if __name__ =="__main__":
  
  _month = loop_month(st="202104", ed="202510")[:6]
  _month = loop_month(st="201904", ed="202510")[:24]
  # _month = ['202104', '202105', '202106', '202107', '202108', '202109']
  
  if 1:
    for month in _month:
      r2_mm(month, isConcatSave=True) #2021.11.10
      
    