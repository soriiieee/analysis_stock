# -*- coding: utf-8 -*-
# when   : 2021.07.15
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
from getErrorValues import me,rmse,mae,r2,prmse,pmae #(x,y)
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
#--------
sys.path.append('/home/ysorimachi/work/hokuriku/py')
from utils import *
#--------
from mapbox import map_lonlat3 #(df,html_path,zoom=4,size_max=None)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

from plot_multi_rad import plot_multi_rad
SMAME_SET3="/work/ysorimachi/hokuriku/dat2/smame/set3_ts" #30min_201912.csv

def load_smame(cate,month):
  path = f"{SMAME_SET3}/{cate}_{month}.csv"
  df = pd.read_csv(path)
  df["time"] = pd.to_datetime(df["time"])
  df = df.set_index("time")
  df["p.u"] = df["sum"]*2/df["max"]
  return df

def get_smame_rad(cate,mm,radname="obs"):
  #teleme ---
  df = load_smame(cate=cate,month=mm)
  # df = get_pu(df)
  #rad ---
  rad = load_rad(month=mm,cate=radname, lag=30)
  rad = rad["mean"]/1000 #W->Kw
  rad.name = "obs"
  df = pd.concat([df,rad],axis=1)
  return df


def max_df(df,cate="max"):
  """
  全カラムがテレメ地点の必要性あり
  """
  _idx = df.index
  _columns = df.columns
  df = df.replace(np.nan,-9999)
  dat = np.where(df.values == -9999,0,1)
  df2 = pd.DataFrame(dat,index=_idx,columns = _columns)
  for c in df2.columns:
    v = teleme_max(code=c,cate=cate)
    df2[c] = df2[c] * v
  return df2

def select(cate= "all",radname="obs",err_name="rmse"):
  #local function --------
  def get_pu(df):
    df2 = max_df(df,cate="max")
    df3 = max_df(df,cate="panel")
    use_col = [ c for c in df.columns if "telm" in c ]
    
    df["sum"] = df[use_col].sum(axis=1)
    df["sum_max"] = df2[use_col].sum(axis=1)
    df["sum_panel"] = df3[use_col].sum(axis=1)
    df["p.u"] = df["sum"]/df["sum_max"]
    return df
  
  def clensing(df,use_col,drop=True):
    n_before = df.shape[0]
    for c in use_col:
      df[c] = df[c].apply(lambda x: np.nan if x<0 or x>1.2 else x)
      
    if drop:
      df = df[use_col]
      df = df.dropna()
    n_after = df.shape[0]
    # print(n_before, "->", n_after)
    return df
  
  def err_func(x):
    
    x1 = x[(x["hhmi"]>600)&(x["hhmi"]<1800)]
    x2 = x[(x["hhmi"]>900)&(x["hhmi"]<1500)]
    
    e1 = rmse(x1["obs"],x1["p.u"])
    e2 = me(x1["obs"],x1["p.u"])
    e3 = pmae(x2["obs"],x2["p.u"])
    e4 = prmse(x2["obs"],x2["p.u"])

    # e1 = rmse(x["p.u"],x["obs"])
    # e2 = me(x["p.u"],x["obs"])
    # e3 = pmae(x["p.u"],x["obs"])
    # e4 = prmse(x["p.u"],x["obs"])
    
    _err = [e1,e2,e3,e4]
    return pd.Series(_err, index=["rmse","me","%mae","%rmse"])
  
  def check_day_err(df):
    df = df.reset_index()
    df["dd"] = df["time"].apply(lambda x: x.strftime("%Y%m%d"))
    
    tmp = df.groupby("dd").apply(err_func)
    tmp = tmp.replace(9999,np.nan)
    tmp = tmp.dropna()
    return tmp
  #local function --------
  
  _mm = loop_month(st="202004")
  _mm19 = loop_month(st="201904")[:12]
  # print(_mm)
  # print(_mm19)
  # sys.exit()
  
  f,ax = plt.subplots(3,4,figsize=(4*5,3*4))
  ax = ax.flatten()
  
  OUTDIR="/home/ysorimachi/work/hokuriku/out/smame/pu/png"
  SELECT="/home/ysorimachi/work/hokuriku/out/smame/pu/select"
  
  _df= []
  for i,(mm,mm19)in enumerate(zip(_mm,_mm19)):
    # mm="202101"
    # dataset
    df = get_smame_rad(cate=cate,mm=mm,radname="obs")
    df = df.dropna().reset_index()
    df["hhmi"] = df["time"].apply(lambda x: int(x.strftime("%H%M")))
    # df1 = get_smame_rad(cate=cate,mm=mm19,radname="obs")
    
    # print(df.head())
    # sys.exit()
    tmp = check_day_err(df)
    _df.append(tmp)
  
  df = pd.concat(_df,axis=0)
  df = df.sort_values(err_name,ascending = False)
  
  df.to_csv(f"{SELECT}/{cate}_worst.csv")
  print("end", cate)
  return

OUTDIR="/home/ysorimachi/work/hokuriku/out/smame/pu/png"
SELECT="/home/ysorimachi/work/hokuriku/out/smame/pu/select"
def check(cate,radname):
  path = f"{SELECT}/{cate}_worst.csv"
  df = pd.read_csv(path)
  df["dd"] = df["dd"].astype(str)
  
  _df =[]
  _dd = df["dd"].values.tolist()[:20]
  _mm = set([ str(dd)[:6] for dd in _dd])
  #----------------
  #list にある月だけ最初に作成しておく---------
  for mm in list(_mm):
    df = get_smame_rad(cate=cate,mm=mm,radname="obs")
    snw = get_snowdepth(mm,"47607") #snowdeta in 
    df = df.merge(snw,on="time",how="inner")
    _df.append(df)
    
  df = pd.concat(_df,axis=0)
  df = df.reset_index()
  df["dd"] = df["time"].apply(lambda x: x.strftime("%Y%m%d"))
  #----------------
  _col = ["obs","p.u"]
  _col2 = ["snowDepth"]
  
  
  # snw = get_snowdepth(mm,"47607")
  
  df2 = df.loc[df["dd"].isin(_dd),:]
  f = plot_multi_rad(df,_dd,_col,_sub_col=_col2,vmin=0,vmax=1)
  f.savefig(f"{SELECT}/{cate}-20_worst.png",bbox_inches="tight")
  df2.to_csv(f"{SELECT}/{cate}-20_worst.csv",index=False)
  
  print(SELECT)
  return 

def snow_error(cate):
  _mm = loop_month(st="202004")
  _dd_snow = []
  for mm in list(_mm):
    # df = get_smame_rad(cate=cate,mm=mm,radname="obs")
    sn = get_snowdepth(mm,"47607") #snowdeta in
    sn["dd"] = sn["time"].apply(lambda x: x.strftime("%Y%m%d"))
    tmp = sn.groupby("dd").agg({"snowDepth":["min","max","mean"]})
    tmp = tmp.reset_index()
    tmp.columns = ["dd","snow(min)","snow(max)","snow(mean)"]
    _dd_snow.append(tmp)
  
  # df = pd.read_csv(f"{SELECT}/{}")
  print(df.head())

if __name__ == "__main__":
  
  err_name="%mae"
  if 0:
    for cate in ["all", "surplus"]:
    # for cate in ["surplus"]:
      select(cate=cate,radname="obs",err_name=err_name) # - make list day
      check(cate=cate,radname="obs") # - plor ts
      # sys.exit()
  
  
  if 1:
    for cate in ["all", "surplus"]:
      snow_error(cate=cate)
      sys.exit()
    
    # sys.exit()
      