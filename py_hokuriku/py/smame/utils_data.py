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
from getErrorValues import me,rmse,mape,r2 #(x,y) pd.Seriees
from convSokuhouData import conv_sfc #(df, ave=minutes,hour)
#amedas relate 2020.02.04 making...
from tool_AMeDaS import code2name, name2code
# from tool_110570 import get_110570,open_110570
# from tool_100571 import get_100571,open_100571
#(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
import subprocess
# outd ='/home/griduser/work/sori-py2/deep/out/0924_01'
# os.makedirs(outd, exist_ok=True)
# subprocess.run('rm -f *.png', cwd=outd, shell=True)
sys.path.append('/home/ysorimachi/work/hokuriku/py')
from utils import *
try:
  from utils_smame import smame_max
except:
  from smame.utils_smame import smame_max
from mapbox import map_lonlat3 #(df,html_path,zoom=4,size_max=None)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

from matplotlib import rcParams
rcParams['font.size'] = 15
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

SMAME_SET3="/work/ysorimachi/hokuriku/dat2/smame/set3_ts" # 30min_201912.csv
SET_DD="/work/ysorimachi/hokuriku/dat2/smame/dataset2"
RAD_DD="/work/ysorimachi/hokuriku/dat2/smame/rad2"

def df_smame(cate,month):
  path = f"{SMAME_SET3}/{cate}_{month}.csv"
  df = pd.read_csv(path)
  df["time"] = pd.to_datetime(df["time"])
  df = df.set_index("time")
  df["p.u"] = df["sum"]*2/df["max"]
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

def load_smame(cate= "all",mm="201904",radname="obs"):
  """[summary] -------------------
  init      : 2021.09.08
  update    : 2021.11.30
  #-------------------------------
  Args:
      cate (str, optional): [description]. Defaults to "all".
      radname (str, optional): [description]. Defaults to "obs".

  Returns:
      [type]: [description]
  """
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
  # ------------------------------------------
  df = df_smame(cate,mm)
  print(df.head())
  sys.exit()
  
  # ------------------------------------------
  return df


OUTDIR="/home/ysorimachi/work/hokuriku/out/smame/pu/png"
def mk_reg_para(cate):
  """ 2021.09.03 sorimachi"""
  # _mm = loop_month(st="202004")
  _mm19 = loop_month(st="201904")[:12]
  
  radname = "obs" 
  # print(_mm)
  # print(_mm19)
  # sys.exit()
  
  # f,ax = plt.subplots(3,4,figsize=(4*5,3*4))
  # ax = ax.flatten()
  
  param={}
  # OUTDIR="/home/ysorimachi/work/hokuriku/out/smame/pu/png"
  # for i,(mm,mm19)in enumerate(zip(_mm,_mm19)):
  for i,mm in enumerate(_mm19):
    # dataset
    df = get_smame_rad(cate=cate,mm=mm,radname=radname)
    # df1 = get_smame_rad(cate=cate,mm=mm19,radname="obs")
    # a0,b0,c0 = get_a_b_c("201905",cate="obs")
    # a1,b1,c1 = get_a_b_c(mm,cate="obs")
    
    # fitting
    df = df[["obs","p.u"]]
    df = df.dropna()
    pf = PolynomialFeatures(degree=3)
    X2 = pf.fit_transform(df["obs"].values.reshape(-1,1))
    lr = LinearRegression().fit(X2,df["p.u"].values)

    _,c,b,a = lr.coef_
    d  = lr.intercept_
    
    param[mm] = [a,b,c,d]
  
  df = pd.DataFrame(param).T
  df.index.name = "month"
  df.columns = ["a","b","c","d"]
  df = df.round(4) #1有効数字をそろえる
  
  outt_path = f"/home/ysorimachi/work/hokuriku/tbl/teleme/coef/ligression_smame_3dim_{radname}.csv"
  df.to_csv(outt_path)
  return

def mk_smame_dd(cate,mm,n_day):
  _path = sorted(glob.glob(f"{SET_DD}/{cate}_{mm}*.csv"))
  rad_path = "../../tbl/smame/list_smame_near_rad.csv"
  df = pd.read_csv(rad_path)
  if cate=="all":
    df =  df[df["cate"]=="Z"]
    # df = df.iloc[:10,:]
    df  = df[["code","near_rad"]].set_index("code")
    near_rad = df.to_dict()["near_rad"]
  
    
  rad  = load_rad(month=mm,cate="8now0", lag=30,only_mean=False)
  rad = rad.reset_index()
  rad["hhmm"] = rad["time"].apply(lambda x: x.strftime("%H%M"))
  for p in tqdm(_path):
    dd = os.path.basename(p).split("_")[1][:8]
    rad2 = rad[rad["dd"]==dd]
    rad2 = rad2.set_index("hhmm").drop(["time"],axis=1)
    time_index = rad2.index
    
    df = pd.read_csv(p)
    df = df.set_index("code")
    
    # print(df.index[:10])
    # sys.exit()
    # df = df.iloc[:20,:]
    rad_hash = {}
    # for code in tqdm(list(df.index)):
    for code in list(df.index):
      # print(code)
      try:
        rad_point = near_rad[code]
      except:
        rad_point ="mean"
      
      # s = rad2[rad_point]
      rad_hash[code] = list(rad2[rad_point])
    
    # rad_sm = pd.concat(_s,axis=1)
    rad_sm = pd.DataFrame(rad_hash).T
    rad_sm.columns = time_index
    
    rad_sm.to_csv(f"{RAD_DD}/{cate}_{dd}.csv")
  return

def premake_mm_data(cate,mm):
  _path = sorted(glob.glob(f"{SET_DD}/{cate}_{mm}*.csv"))
  _rad_path = sorted(glob.glob(f"{RAD_DD}/{cate}_{mm}*.csv"))
  
  def calc_time(x,dd):
    x = str(x).zfill(4)
    hh,mi = map(int, [x[:2],x[2:]])
    yy,mm,d2 = map(int, [dd[:4],dd[4:6],dd[6:8]])
    
    if hh !=24:
      time = datetime(yy,mm,d2,hh,mi)
    else:
      time = datetime(yy,mm,d2,0,0) + timedelta(days=1)
    return time
  
  _df =[]
  for (p,r_path) in tqdm(list(zip(_path,_rad_path))):
    
    # --pv --
    dd = os.path.basename(p).split("_")[1][:8]
    df = pd.read_csv(p).set_index("code")
    df = df.replace(9999,np.nan)
    
    df2 = df.T
    _max = []
    for i,r in df2.iterrows():
      _code = list(r.dropna().index)
      max_pv = smame_max(_code)

      _max.append(max_pv)
    # df.sum()
    # --rad --
    rad = pd.read_csv(r_path).set_index("Unnamed: 0")
    rad.index.name = "code"
    if "0000" in rad.columns:
      rad = rad.drop("0000",axis=1)
    if not "2400" in rad.columns:
      rad["2400"] = np.nan
    
    s1 = df.mean()
    s2 = df.sum()
    s3 = df.std()
    s4 = df.count()
    # s5 = pd.Series(np.array(_max))
    r1 = rad.mean()
    
    df = pd.concat([s1,s2,s3,s4,r1],axis=1)
    df.index.name = "hhmm"
    df.columns = ["mean","sum","std","count","rad"]
    df = df.reset_index()
    df["time"] = df["hhmm"].apply(lambda x: calc_time(x,dd))
    df = df.set_index("time")
    df["max_pv"] = _max    
    # print(df.head())
    # sys.exit()
    _df.append(df)
  
  df = pd.concat(_df,axis=0)
  df.to_csv(f"{SMAME_SET3}/{cate}_{mm}.csv")
  return 

def laod_smame_with_rad(cate,mm):
  path = f"{SMAME_SET3}/{cate}_{mm}.csv"
  df = pd.read_csv(path)
  df["time"] = pd.to_datetime(df["time"])
  df = df.set_index("time")
  df["p.u_sm"] = df["sum"] / df["max_pv"] *2
  return df

  # _path= [ pd.read_csv(f) for f in sorted(glob.glob(f"{RAD_DD}/{cate}_{mm}*.csv"))]
  
# def laod_smame_with_rad(cate,mm):
#   path = f"{SMAME_SET3}/{cate}_{mm}.csv"
#   df = pd.read_csv(path)
#   df["time"] = pd.to_datetime(df["time"])
#   df = df.set_index("time")
#   return df



if __name__ == "__main__":
  
  if 0:
    # -------------------------
    _mm = loop_month()[:24]
    _dd = loop_day()[:24]
    # print(_dd)
    # sys.exit()
    # -------------------------
    for mm,dd in zip(_mm,_dd):
      # mm="201905"
      mk_smame_dd(cate="all",mm=mm,n_day=dd) #smame dd - data...
      premake_mm_data(cate="all",mm=mm) # premake dataset ...
      print(datetime.now(),"[END]", mm) 
      # sys.exit()
      
  if 1:
    get_data("201904","1200",cate="all")

      
      