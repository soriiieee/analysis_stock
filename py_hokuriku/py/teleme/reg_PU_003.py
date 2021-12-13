# -*- coding: utf-8 -*-
# when   : 2021.09.15 "時刻別回帰の回帰係数の策定"
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
from getErrorValues import me,rmse,mae,r2,mape #(x,y)
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
try:
  from utils_teleme import *
  from utils_data import teleme_and_rad
except:
  from teleme.utils_teleme import *
  from utils_data import teleme_and_rad
from utils import *
from smame.utils_data import laod_smame_with_rad #(cate,mm)

from mapbox import map_lonlat3 #(df,html_path,zoom=4,size_max=None)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures


from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 18
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']


TELEME="/work/ysorimachi/hokuriku/dat2/teleme/dataset" #30min_201912.csv

def load_teleme(month,min=30):
  path = f"{TELEME}/{min}min_{month}.csv"
  df = pd.read_csv(path)
  df["time"] = pd.to_datetime(df["time"])
  df = df.set_index("time")
  # df.to_csv("/home/ysorimachi/work/hokuriku/dat/rad/r2/rad/tmp_check2.csv")
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

OUTDIR="/home/ysorimachi/work/hokuriku/out/teleme/pu/png"


def fit_PU(df,xy=["obs","p.u"],degree=2):
  """
  2021.09.16 -> 変数を返すようなprgram
  """
  df = df[["p.u","obs"]]
  df = df.dropna()
  xc,yc = xy[:]
    # print(mm,df.shape)
    # continue
    # sys.exit()
  pf = PolynomialFeatures(degree=degree)
  X2 = pf.fit_transform(df[xc].values.reshape(-1,1))
  lr = LinearRegression().fit(X2,df[yc].values)
  
  if degree==2:
    _,b,a = lr.coef_
    c  = lr.intercept_
    
    # a = np.round(a,4)
    # b = np.round(b,4)
    # c = np.round(c,4)
    return [a,b,c]
  if degree==3:
    _,c,b,a = lr.coef_
    d = lr.intercept_
    return [a,b,c,d]

def list_hh(st="0600",ed="1800"):
  _hh = pd.date_range(start=f"20210101{st}", end=f"20210101{ed}", freq="30T")
  _hh = [t.strftime("%H%M") for t in _hh]
  return _hh

OUT_HOME="/home/ysorimachi/work/hokuriku/out/teleme/pu_hh"
def reg_pu_mm_hh(m_shift=1, h_shift=1,radname="8now0",select="all",with_smame=False):
  
  """[月別-時刻別回帰式のパラメータ作成]
  2021.09.15 月別の回帰係数を実施してみる
  2021.11.10 更新(地点数を選択可能)
    *全学習期間のデータを取得
    *時刻と月は前後1コマでデータ数確保
    *今後は、快晴指数別に分岐するprogramを作成予定(2022年作業予定)
  Returns:
    None
      [type]: [description]
  """
  # subroutine functions --------------
  def set_data(use_mm, use_hh):
    # _df = [ teleme_and_rad(radname=radname,mm=mm, select=select) for mm in use_mm]
    _df=[]
    teleme_col=["p.u","obs","sum","sum_max"]
    for mm in use_mm:
      df = teleme_and_rad(radname=radname, mm=mm,select="all")[teleme_col]
      df = clensing_col(df,_col = teleme_col) #clensing
      
      #------------------------------------------------------
      if with_smame:
        # smame (all) ---
        smame_col = ["p.u_sm","rad","sum","max_pv"]
        sm = laod_smame_with_rad(cate="all",mm=mm)[smame_col]
        sm = clensing_col(sm,_col = smame_col) #clensing
        sm = sm.rename(columns = {"sum": "sum_sm"})
        # -concat(teleme + smame[All])------------------
        df = pd.concat([df,sm],axis=1)
        # pd.concat([,],axis=1)
        df["sum"] += df["sum_sm"]
        df["sum_max"] += df["max_pv"]/2
        df["p.u"] = df["sum"] / df["sum_max"]
        df = df.drop(["p.u_sm","rad","sum_sm","max_pv"],axis=1)
      #------------------------------------------------------
      _df.append(df)
      #-----------------------
    
    # _df = [ estimate(radname="8now0",CSV=False,ONLY_DATA=True,mm=mm) for mm in tqdm(use_mm)]
    df = pd.concat(_df,axis=0)
    df = df.reset_index()
    df["hh"] = df["time"].apply(lambda x: x.strftime("%H%M"))
    df = df.loc[df["hh"].isin(use_hh),:]
    return df

  def list_shift(mm,list_month,n=1):
    _mm = [mm]
    idx = list_month.index(mm)
    # print(list_month, idx)
    list2 = np.roll(list_month,-idx)
    # print(list2)
    return np.roll(list2,n)[0]
  # subroutine functions --------------
  #----------------mai
  
  # df use ---
  _month = loop_month()[:12]
  
  params={}
  for mm in _month[:]:
    # mm="201905"
    m2 = list_shift(mm,_month,n=-1)
    m1 = list_shift(mm,_month,n=1)
    use_mm = [m1,mm,m2]
    
    _hh = list_hh(st="0530",ed="1830")
    for hh in tqdm(_hh[1:-1][:]):
      h1 = list_shift(hh,list_hh(),n=1)
      h2 = list_shift(hh,list_hh(),n=-1)
      use_hh = [h1,hh,h2]
      #-- make dataset
      df = set_data(use_mm, use_hh)
      a,b,c = fit_PU(df,xy=["obs","p.u"],degree=2)
      # print(a,b,c,df.dropna(subset=["p.u","obs"]).shape)
      params[f"{mm}_{hh}"] = [a,b,c]
      #df save
      # df.to_csv(f"{OUT_HOME}/csv/df/set_mm{mm}_hh{hh}.csv", index=False)
      # print(datetime.now(),"[END]",use_mm,"-",use_hh)
  
  #params --
  df = pd.DataFrame(params).T
  df.columns = ["a","b","c"]
  _mm = [ x.split("_")[0] for x in df.index]
  _hh = [ x.split("_")[1] for x in df.index]
  df["mm"] = _mm
  df["hh"] = _hh
  
  if with_smame:
    outtpath = f"{OUT_HOME}/param/regPara_m{m_shift}_h{h_shift}_sm.csv"
  else:
    outtpath = f"{OUT_HOME}/param/regPara_m{m_shift}_h{h_shift}.csv"
    
  df.to_csv(outtpath, index=False)
  print("reg_pu_mm_hh()->",outtpath)
  # print(f"{OUT_HOME}/param/regPara_m{m_shift}_h{h_shift}.csv")
  return

def reg_pu_hh(radname="8now0",with_smame=False):
  """
  2021.10.03 実施で、時刻別の回帰式を作成する様にする
  2021.11.10 でーた確保部分修正の実施
  """
  _mm = loop_month()[:12]
  # _df = [ estimate(radname=radname,CSV=False,ONLY_DATA=True,mm=mm) for mm in _mm]
  
  _df=[]
  teleme_col=["p.u","obs","sum","sum_max"]
  for mm in _mm:
    df = teleme_and_rad(radname=radname, mm=mm,select="all")[teleme_col]
    df = clensing_col(df,_col = teleme_col) #clensing
    
    #------------------------------------------------------
    if with_smame:
      # smame (all) ---
      smame_col = ["p.u_sm","rad","sum","max_pv"]
      sm = laod_smame_with_rad(cate="all",mm=mm)[smame_col]
      sm = clensing_col(sm,_col = smame_col) #clensing
      sm = sm.rename(columns = {"sum": "sum_sm"})
      # -concat(teleme + smame[All])------------------
      df = pd.concat([df,sm],axis=1)
      # pd.concat([,],axis=1)
      df["sum"] += df["sum_sm"]
      df["sum_max"] += df["max_pv"]/2
      df["p.u"] = df["sum"] / df["sum_max"]
      df = df.drop(["p.u_sm","rad","sum_sm","max_pv"],axis=1)
    #------------------------------------------------------
    _df.append(df)
    #-----------------------
  
  
  df = pd.concat(_df,axis=0)
  df = df.dropna(subset=["obs"])
  df = df.reset_index()
  df["time"] = pd.to_datetime(df["time"])
  df["hh"] = df["time"].apply(lambda x: x.strftime("%H%M"))
  use_col = ["hh","p.u","obs"]
  df = df[use_col]
  _hh = sorted(df["hh"].unique().tolist())
  
  param={}
  for hh in _hh:
    tmp = df[df["hh"]==hh]
    a,b,c = fit_PU(tmp)
    
    param[hh] = [a,b,c]
  
  df = pd.DataFrame(param).T
  df.index.name = "hh"
  df.columns = ["a","b","c"]
  
  if with_smame:
    path = f"{OUT_HOME}/param/regParam_hh_2019_sm.csv"
  else:
    path = f"{OUT_HOME}/param/regParam_hh_2019.csv"
  df.to_csv(path)
  print("reg_pu_hh() ->",path)
  return
  
def reg_pu_mm(radname="8now0",with_smame=False):
  """
  2021.09.2
  2021.11.09 update
  8now0日射量にして、回帰式を再度検証する
  """
  _mm19 = loop_month(st="201904")[:12]
  
  param = {}
  # for mm in tqdm(_mm19):
  for mm in tqdm(_mm19):
    # mm="202003"
    # df = estimate(radname=radname,ONLY_DATA=True,mm=mm)
    # teleme --------------------
    teleme_col=["p.u","obs","sum","sum_max"]
    df = teleme_and_rad(radname=radname, mm=mm,select="all")[teleme_col]
    df = clensing_col(df,_col = teleme_col) #clensing
    
    
    if with_smame:
      # smame (all) ---
      smame_col = ["p.u_sm","rad","sum","max_pv"]
      sm = laod_smame_with_rad(cate="all",mm=mm)[smame_col]
      sm = clensing_col(sm,_col = smame_col) #clensing
      sm = sm.rename(columns = {"sum": "sum_sm"})
      # -concat(teleme + smame[All])------------------
      df = pd.concat([df,sm],axis=1)
      # pd.concat([,],axis=1)
      df["sum"] += df["sum_sm"]
      df["sum_max"] += df["max_pv"]/2
      df["p.u"] = df["sum"] / df["sum_max"]
      
    # fitting
    df = df[["p.u","obs"]]
    # print(df.head(50))
    # sys.exit()
    # print(df.shape)
    df = df.dropna()
    # print(mm,df.shape)
    # continue
    # sys.exit()
    pf = PolynomialFeatures(degree=2)
    X2 = pf.fit_transform(df["obs"].values.reshape(-1,1))
    lr = LinearRegression().fit(X2,df["p.u"].values)
    
    _,b,a = lr.coef_
    c  = lr.intercept_
    param[mm] = [a,b,c]
  
  df = pd.DataFrame(param).T
  df.index.name = "month"
  df.columns = ["a","b","c"]
  df = df.round(4) #1有効数字をそろえる
  
  if with_smame:
    outt_path = f"/home/ysorimachi/work/hokuriku/tbl/teleme/coef/ligression_values_{radname}_sm.csv"
  else:
    outt_path = f"/home/ysorimachi/work/hokuriku/tbl/teleme/coef/ligression_values_{radname}.csv"
  df.to_csv(outt_path)
  print("reg_pu_mm() ->",outt_path)
  return
    

if __name__ == "__main__":
  
  
  if 0:
    """ 
    2019年度で、回帰式を作成 -> 時刻別回帰式の策定 
    月別回帰や時刻別/月別・時刻別等の切り口も併せて表示する
    """
    radname="8now0"
    reg_pu_mm(radname=radname) #月別回帰
    reg_pu_hh(radname=radname) #時刻別回帰
    reg_pu_mm_hh(m_shift=1, h_shift=1,radname=radname) #月別/時刻別回帰
    #--------------
    # reg_pu_mm() -> /home/ysorimachi/work/hokuriku/tbl/teleme/coef/ligression_values_8now0.csv
    # reg_pu_hh() -> /home/ysorimachi/work/hokuriku/out/teleme/pu_hh/param/regParam_hh_2019.csv
    # reg_pu_mm_hh()-> /home/ysorimachi/work/hokuriku/out/teleme/pu_hh/param/regPara_m1_h1.csv に保存されているので、ココカラcheckしていけばよい　a,b,cの回帰係数を算出()
  if 1:
    """ 
    時刻別回帰式の策定 (With SMAME-All )
    """
    radname="8now0"
    reg_pu_mm(radname=radname,with_smame=True) #月別回帰
    # reg_pu_hh(radname=radname,with_smame=True) #時刻別回帰
    # reg_pu_mm_hh(m_shift=1, h_shift=1,radname=radname,with_smame=True) #月別/時刻別回帰

  
  if 0:
    for err_name in ["ME","RMSE","%MAE"]:
      seido_plot(err_name=err_name) #回帰式/
      print("end", err_name)
      # sys.exit()