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
from getErrorValues import me,rmse,mae,r2 #(x,y)
from convSokuhouData import conv_sfc #(df, ave=minutes,hour)
#amedas relate 2020.02.04 making...
from tool_AMeDaS import code2name, name2code
from tool_110570 import get_110570,open_110570
from tool_100571 import get_100571,open_100571
#(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
import subprocess
# outd ='/home/griduser/work/sori-py2/deep/out/0924_01'
# os.makedirs(outd, exist_ok=True)
# subprocess.run('rm -f *.png', cwd=outd, shell=True)
sys.path.append('/home/ysorimachi/work/hokuriku/py')
from utils import *
from utils_smame import fit_PU
from mapbox import map_lonlat3 #(df,html_path,zoom=4,size_max=None)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

from matplotlib import rcParams
rcParams['font.size'] = 15
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

SMAME_SET3="/work/ysorimachi/hokuriku/dat2/smame/set3_ts" # 30min_201912.csv

def load_smame(cate,month):
  path = f"{SMAME_SET3}/{cate}_{month}.csv"
  df = pd.read_csv(path)
  df["time"] = pd.to_datetime(df["time"])
  df = df.set_index("time")
  df["p.u"] = df["sum"]*2/df["max"]
  return df

def laod_smame_with_rad(cate,mm):
  path = f"{SMAME_SET3}/{cate}_{mm}.csv"
  df = pd.read_csv(path)
  df["time"] = pd.to_datetime(df["time"])
  df = df.set_index("time")
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
  # df = load_smame(cate=cate,month=mm)
  # df = get_pu(df)
  #rad ---
  rad = load_rad(month=mm,cate=radname, lag=30)
  rad = rad["mean"]/1000 #W->Kw
  rad.name = "obs"
  # df = pd.concat([df,rad],axis=1)
  return rad


OUTDIR="/home/ysorimachi/work/hokuriku/out/smame/pu/png"
def reg_mm(cate="all",radname="obs"):
  """[summary]
    2021.09.03 init dataset!
    2021.11.30 update ... smamedata with 8Now0 rad data
  Args:
      cate (str, optional): [description]. Defaults to "all".
      radname (str, optional): [description]. Defaults to "obs".
  Return:
    None(csv-params save !)
  """
  _mm = loop_month()[:12]
  # _mm19 = loop_month()
  param={}
  # OUTDIR="/home/ysorimachi/work/hokuriku/out/smame/pu/png"
  # for i,(mm,mm19)in enumerate(zip(_mm,_mm19)):
  for (i,mm) in tqdm(list(enumerate(_mm))):
    # dataset
    df = laod_smame_with_rad(cate=cate,mm=mm)
    # print(mm,df.shape)
    # continue
    if radname == "obs":
      rad = get_smame_rad(cate=cate,mm=mm,radname=radname)
      df = pd.concat([df,rad],axis=1)
    else:
      df = df.rename(columns={"rad":"obs"})
      df["obs"] /=1000
      
    df = df.dropna(subset=["max_pv"])
    df["p.u"] = df["sum"] / (df["max_pv"] / 2. )
    # fitting
    df = df[["obs","p.u"]]
    df = df.dropna()
    # print(mm,df.shape)
    # continue
    # print(df.head())
    # print(df.describe())
    # sys.exit()
    pf = PolynomialFeatures(degree=2)
    X2 = pf.fit_transform(df["obs"].values.reshape(-1,1))
    # print(X2)
    # sys.exit()
    lr = LinearRegression().fit(X2,df["p.u"].values)

    _,b,a = lr.coef_
    c  = lr.intercept_
    
    mm2 = mm[4:6]
    param[mm2] = [a,b,c]
  
  df = pd.DataFrame(param).T
  df.index.name = "month"
  df.columns = ["a","b","c"]
  df = df.round(4) #1有効数字をそろえる
  
  outt_path = f"/home/ysorimachi/work/hokuriku/tbl/smame/coef/param_mm_{radname}.csv"
  df.to_csv(outt_path)
  print(outt_path)
  return

def reg_hh(cate="all",radname="obs"):
  """[summary]
    2021.09.03 init dataset!
    2021.11.30 update ... smamedata with 8Now0 rad data
  Args:
      cate (str, optional): [description]. Defaults to "all".
      radname (str, optional): [description]. Defaults to "obs".
  Return:
    None(csv-params save !)
  """
  _mm = loop_month()[:12]
  # _mm19 = loop_month()
  # OUTDIR="/home/ysorimachi/work/hokuriku/out/smame/pu/png"
  # for i,(mm,mm19)in enumerate(zip(_mm,_mm19)):
  _df = []
  for (i,mm) in tqdm(list(enumerate(_mm))):
    # dataset
    df = laod_smame_with_rad(cate=cate,mm=mm)
    # print(mm,df.shape)
    # continue
    if radname == "obs":
      rad = get_smame_rad(cate=cate,mm=mm,radname=radname)
      df = pd.concat([df,rad],axis=1)
    else:
      df = df.rename(columns={"rad":"obs"})
      df["obs"] /=1000
      
    df = df.dropna(subset=["hhmm"])
    _df.append(df)
    
  df = pd.concat(_df,axis=0)
  df["hhmm"] = df["hhmm"].apply(lambda x: str(int(x)).zfill(4) )
  
  _hh = sorted(df["hhmm"].unique().tolist())
  #------------------------
  
  param={}
  for hh in _hh:
    tmp = df[df["hhmm"]==hh]
    tmp = tmp.dropna(subset=["max_pv"])
    tmp["p.u"] = tmp["sum"] / (tmp["max_pv"] / 2. )
    # fitting
    tmp = tmp[["obs","p.u"]]
    tmp = tmp.dropna()
    
    if tmp.shape[0]>10:
      
      pf = PolynomialFeatures(degree=2)
      X2 = pf.fit_transform(tmp["obs"].values.reshape(-1,1))
    # print(X2)
    # sys.exit()
      lr = LinearRegression().fit(X2,tmp["p.u"].values)

      _,b,a = lr.coef_
      c  = lr.intercept_
      param[hh] = [a,b,c]
    else:
      param[hh] = [ 9999.  ,9999.  ,9999.  ]

    # print(hh)
    # sys.exit()
  df = pd.DataFrame(param).T
  df.index.name = "hhmm"
  df.columns = ["a","b","c"]
  df = df.round(4) #1有効数字をそろえる
  
  outt_path = f"/home/ysorimachi/work/hokuriku/tbl/smame/coef/param_hh_{radname}.csv"
  df.to_csv(outt_path)
  print(outt_path)
  return

def list_hh(st="0600",ed="1800"):
  _hh = pd.date_range(start=f"20210101{st}", end=f"20210101{ed}", freq="30T")
  _hh = [t.strftime("%H%M") for t in _hh]
  return _hh

def reg_mm_hh(cate="all",radname="obs",m_shift=1,h_shift=1):
  """[summary]

  Args:
      cate (str, optional): [description]. Defaults to "all".
      radname (str, optional): [description]. Defaults to "obs".
  Return:
      None
  """
  _mm = loop_month()[:12]
  # _mm19 = loop_month()
  # OUTDIR="/home/ysorimachi/work/hokuriku/out/smame/pu/png"
  # for i,(mm,mm19)in enumerate(zip(_mm,_mm19)):
  _df = []
  for (i,mm) in tqdm(list(enumerate(_mm))):
    # dataset
    df = laod_smame_with_rad(cate=cate,mm=mm)
    if radname == "obs":
      rad = get_smame_rad(cate=cate,mm=mm,radname=radname)
      df = pd.concat([df,rad],axis=1)
    else:
      df = df.rename(columns={"rad":"obs"})
      df["obs"] /=1000
    
    df = df.dropna(subset=["hhmm"])
    _df.append(df)
  df = pd.concat(_df,axis=0)
  df["hhmm"] = df["hhmm"].apply(lambda x: str(int(x)).zfill(4) )
  df = df.reset_index()
  df["mm"] = df["time"].apply(lambda x: x.strftime("%Y%m"))
  _hh = sorted(df["hhmm"].unique().tolist())
  df = df.set_index("time")
  #------------------------

  def list_shift(mm,list_month,n=1):
    _mm = [mm]
    idx = list_month.index(mm)
    # print(list_month, idx)
    list2 = np.roll(list_month,-idx)
    # print(list2)
    return np.roll(list2,n)[0]
  
  def set_data(df,use_mm, use_hh):
    df = df.loc[df["mm"].isin(use_mm),:]
    df = df.loc[df["hhmm"].isin(use_hh),:]
    
    return df
  # subroutine functions --------------
  #----------------mai
  
  # df use ---
  _month = loop_month()[:12]
  # _hh = list_hh(st="0530",ed="1830")
  params={}
  for mm in tqdm(_month[:]):
    # mm="201905"
    m2 = list_shift(mm,_month,n=-m_shift)
    m1 = list_shift(mm,_month,n=m_shift)
    use_mm = [m1,mm,m2]
        
    _hh = list_hh(st="0530",ed="1830")
    for hh in tqdm(_hh[1:-1][:]):
      h1 = list_shift(hh,_hh,n=h_shift)
      h2 = list_shift(hh,_hh,n=-h_shift)
      use_hh = [h1,hh,h2]
      #-- make dataset
      # print(df.shape)
      tmp = set_data(df,use_mm, use_hh)
      # print(tmp.shape)
      # sys.exit()
      tmp["p.u"] = tmp["sum"] / (tmp["max_pv"] / 2. )
    # fitting
      tmp = tmp[["obs","p.u"]]
      tmp = tmp.dropna()
      if tmp.shape[0] >10:
        a,b,c = fit_PU(tmp,xy=["obs","p.u"],degree=2)
      else:
        a,b,c = 9999.,9999.,9999.
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
  df.to_csv(f"/home/ysorimachi/work/hokuriku/tbl/smame/coef/param_m{m_shift}_h{h_shift}_{radname}.csv", index=False)
  print(f"/home/ysorimachi/work/hokuriku/tbl/smame/coef/param_m{m_shift}_h{h_shift}_{radname}.csv")
  return

#-----------------------------------------------------------
def get_a_b_c_d(month,cate):
  mm=month[4:6]
  if cate == "obs":
    path = "/home/ysorimachi/work/hokuriku/tbl/teleme/coef/ligression_smame_3dim_obs.csv"
  if cate == "8now0":
    """sorimachi making 2021.09.02 """
    path = "/home/ysorimachi/work/hokuriku/tbl/teleme/coef/ligression_smame_3dim_8now0.csv"
    
  df = pd.read_csv(path)
  df["mm"] = df["month"].astype(str).apply(lambda x: x[4:6])
  df = df[["mm","a","b","c","d"]].set_index("mm").T
  para = df.to_dict()
  return para[mm]["a"],para[mm]["b"],para[mm]["c"],para[mm]["d"]


def plot_pu_line():
  """ 2021.09.02 """
  f,ax = plt.subplots(3,4,figsize=(4*5,3*4))
  ax = ax.flatten()
  _mm19 = loop_month(st="201904")[:12]
  rcParams['font.size'] = 14
  
  for i,mm in enumerate(_mm19):
    
    # line
    X_line = np.linspace(0,1,1000)
    _label = ["19年05月(unyo)","19年05月(8now0)","月別(unyo)","月別(8now0)"]
    
    #scatter
    df = get_smame_rad(cate="surplus",mm=mm,radname="8now0")
    df = df.dropna()
    ax[i].scatter(df["obs"],df["p.u"],s=2,color="k",alpha=0.8)
    # print(df.describe())
    # sys.exit()
    
    
    for j ,lbl in enumerate(_label):
      
      if j==0:
        month, cate= "201905","obs"
        color,alpha,ls= "gray",0.8,"--"
      if j==1:
        month, cate= "201905","8now0"
        color,alpha,ls= "gray",0.8,"-."
      if j==2:
        month, cate= mm,"obs"
        color,alpha,ls= "blue",1,"-"
      if j==3:
        month, cate= mm,"8now0"
        color,alpha,ls= "red",1,"-"
        
      a,b,c,d = get_a_b_c_d(month,cate=cate)
      
      # pu_pred0 = a*X_line**2 + b*X_line + c
      pu_pred0 = a*X_line**3 + b*X_line**2 + c*X_line + d
      ax[i].plot(X_line,pu_pred0,lw=1,color=color, label=lbl, alpha=alpha, linestyle = ls)
      # pu_pred1 = a1*X_line**2 + b1*X_line + c1
      # pu_pred2 = lr.predict(pf.fit_transform(X_line.reshape(-1,1)))
    # if 1:
    #   ax[i].plot(X_line,pu_pred0,lw=1,color="k", label="201905")
    #   ax[i].plot(X_line,pu_pred1,lw=1,color="b",label="per-Month")
    #   ax[i].plot(X_line,pu_pred2,lw=1,color="r", label="2020-ver")
    
    # ax[i].legend(loc="lower right",fontsize=8)
    ax[i].legend(loc="upper left",fontsize=8)
    ax[i].set_title(f"{mm}[smame(余剰)]")
    
    ax[i].set_ylim(0,1)
    ax[i].set_xlim(0,1)
    ax[i].set_ylabel("p.u.")
    ax[i].set_xlabel("平均日射量[kW/m2]")
    print(datetime.now(),"[end]", mm)
    # sys.exit()
    #---------------------------------------------------------
  plt.subplots_adjust(wspace=0.5, hspace=0.5)
  f.savefig(f"{OUTDIR}/line_pu_2rad_all.png",bbox_inches="tight")
  print(OUTDIR)




if __name__ == "__main__":
  
  if 1:
    # for radname in ["obs","8now0"]:
    for radname in ["8now0"]:
      # reg_mm(cate="all",radname = radname)
      # reg_hh(cate="all",radname = radname)
      # reg_mm_hh(cate="all",radname = radname,m_shift=1,h_shift=1)
    # plot_pu_line()
    sys.exit()
    
  if 1:
    for cate in ["all"]:
    # for cate in ["surplus"]:
      estimate(cate=cate)