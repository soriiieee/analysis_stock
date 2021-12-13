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
from getErrorValues import me,rmse,mape,r2 #(x,y)
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
from utils_smame import fit_PU, get_mm_abc_Z, get_hh_abc_Z,get_mm_hh_abc_Z,get_smame_hosei,smame_max_code
dict_sm = smame_max_code()

from utils_data import laod_smame_with_rad #(cate,mm)
# ---------------------------------------------------

from mapbox import map_lonlat3 #(df,html_path,zoom=4,size_max=None)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

from matplotlib import rcParams
rcParams['font.size'] = 15
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

SMAME_SET3="/work/ysorimachi/hokuriku/dat2/smame/set3_ts" # 30min_201912.csv
ESTIMATE="/home/ysorimachi/work/hokuriku/out/smame/pu/estimate"
RAD_DD="/work/ysorimachi/hokuriku/dat2/smame/rad2"
SET_DD="/work/ysorimachi/hokuriku/dat2/smame/dataset2"
ERR="/home/ysorimachi/work/hokuriku/out/smame/pu/err"

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


def calc_mm_PV(df,mm):
  
  for mtd in [0,1,2]:
    if mtd==0:
      a,b,c = get_mm_abc_Z("05", "obs")
      hosei = 1
    if mtd==1:
      a,b,c = get_mm_abc_Z("05", "obs")
      hosei = get_smame_hosei(cate="all",mm=mm[4:6])
    if mtd ==2:
      a,b,c = get_mm_abc_Z(mm[4:6], "8now0")
      hosei = 1
    
    df[f"PV-max[{mtd}]"] = df[["rad","max_pv"]].apply(lambda x: 
      (a*x[0]**2 + b*x[0]+c)*x[1] * hosei /2.
      ,axis=1)
  return df

def calc_hh_PV(df,hh):
  # mm = str(mm)[4:6]
  hh = str(hh).zfill(4)
  a,b,c = get_hh_abc_Z(hh, "8now0")
  if a==9999:
    df[f"PV-max[3]"] = 9999
  else:
    df[f"PV-max[3]"] = df[["rad","max_pv"]].apply(lambda x: 
      (a*x[0]**2 + b*x[0]+c)*x[1] /2.
      ,axis=1)
  return df

def calc_mm_hh_PV(df,mm,hh):
  mm2 = str(mm)[4:6]
  hh = str(hh).zfill(4)
  a,b,c = get_mm_hh_abc_Z(mm2,hh, radname="8now0")
  if a==9999:
    df[f"PV-max[4]"] = 9999
  else:
    df[f"PV-max[4]"] = df[["rad","max_pv"]].apply(lambda x: 
      (a*x[0]**2 + b*x[0]+c)*x[1] /2.
      ,axis=1)
  return df

def calc_mm_hh_PV_mesh(df,mm,hh):
  mm2 = str(mm)[4:6]
  hh = str(hh).zfill(4)
  
  #-------------------------------
  a,b,c = get_mm_hh_abc_Z(mm2,hh, radname="8now0")
  if a == 9999:
    print(mm,hh,"Null!")
    df[f"PV-max[4]"] = 9999
  else:
    cate="all"
    # print(dict_sm.head())

    _t,_v = [],[]
    _rad = sorted(glob.glob(f"{RAD_DD}/{cate}_{mm}*.csv"))
    _set = sorted(glob.glob(f"{SET_DD}/{cate}_{mm}*.csv"))

    for (rad_path,set_path) in tqdm(list(zip(_rad,_set))):
      dd = os.path.basename(rad_path).split("_")[1][:8]
      ini_j = pd.to_datetime(f"{dd}/{hh}")
      _t.append(ini_j)
      
      rad = pd.read_csv(rad_path).set_index("Unnamed: 0")
      # sm = pd.read_csv(set_path).set_index("code")
      # print(sm.T.isnull().sum())
      # sys.exit()
      rad.index.name = "code"
      rad = rad.reset_index()
      rad["max_pv"] = rad["code"].apply(lambda x: dict_sm[x] if x in list(dict_sm.keys()) else np.nan)
      rad[hh] /=1000
      rad = rad[[hh,"max_pv"]].dropna()
      
      v = ((a*rad[hh]**2 + b*rad[hh]+c)*rad["max_pv"] /2.).sum()
      _v.append(v)
      
    tmp = pd.DataFrame()
    tmp["time"] = _t
    tmp["PV-max[4]"] = _v
    
    df  = df.merge(tmp, on="time", how="left")
    # df[f"PV-max[4]"] = df[["rad","max_pv"]].apply(lambda x: 
    #   (a*x[0]**2 + b*x[0]+c)*x[1] /2.
    #   ,axis=1)
  return df


def estimate(cate="all",seido="train",radname="8now0"):
  """[summary]
  複数層別化を行った場合の推定PVの算出
  Args:
      cate (str, optional): [description]. Defaults to "all".
      seido (str, optional): [description]. Defaults to "train".
      radname (str, optional): [description]. Defaults to "8now0".

  Returns:
      [type]: [description]
  """
  
  if seido == "train":
    _mm = loop_month()[:12]
  else:
    _mm = loop_month()[12:24]
  # ---------------------------------------
  def preprocess(df):
    df = df.reset_index()
    df["hh"] = df["hhmm"].apply(lambda x: str(x).zfill(4))
    df["mm"] = df["time"].apply(lambda x: x.strftime("%m"))
    df["month"] = df["time"].apply(lambda x: x.strftime("%Y%m"))
    df = df.dropna(subset=["rad"])
    df["rad"] /=1000 #W->kW
    return df
  # -----------------------------------------
  _df = [ laod_smame_with_rad(cate,mm) for mm in _mm ]  
  df = pd.concat(_df,axis=0)
  df = preprocess(df)
  
  _df=[]

  for mm in tqdm(_mm):
    tmp = df[df["month"]==mm]
    tmp = calc_mm_PV(tmp,mm)
    
    _hh = sorted(df["hhmm"].unique().tolist())
    for hh in _hh:
      tmp2 = tmp[tmp["hhmm"]==hh]
      
      tmp2 = calc_hh_PV(tmp2,hh)
      tmp2 = calc_mm_hh_PV(tmp2,mm,hh)
      # tmp2 = calc_mm_hh_PV_mesh(tmp2,mm,hh)
      _df.append(tmp2)
    # sys.exit()
  
  df = pd.concat(_df,axis=0)
  df = df.sort_values("time")
  df = df.set_index("time")
  df.to_csv(f"{ESTIMATE}/PV_{seido}_{cate}_{radname}.csv")
  print(datetime.now(),"[END]", f"{ESTIMATE}/PV_{seido}_{cate}_{radname}.csv")
  return 

def get_err(df,tc,pc):
  """ 2021.09.02 """
  """ 2021.11.11 """
  df = df.reset_index()
  df["hh"] = df["time"].apply(lambda x: x.hour)
  
  if df.shape[0] > 5:
    # sys.exit()
    d1 = df[(df["hh"]>=6)&(df["hh"]<=18)]
    d2 = df[(df["hh"]>=9)&(df["hh"]<=15)]
    
    e1 = me(d1[tc],d1[pc])
    # e1 = me(d1["PV-max"],d1["sum"])
    e2 = rmse(d1[tc],d1[pc])
    e3 = mape(d2[tc],d2[pc])
    
    e4 = rmse(d1[tc],d1[pc]) / np.mean(d1[tc])
  else:
    e1,e2,e3,e4 = 9999,9999,9999,9999
  return e1,e2,e3,e4

def calc_mm_seido(cate,seido,radname):
  path = f"{ESTIMATE}/PV_{seido}_{cate}_{radname}.csv"
  df = pd.read_csv(path)
  df["time"] = pd.to_datetime(df["time"])
  df = df.set_index("time")
  # print(df.head(10))
  # sys.exit()
  
  pv_col = [ c for c in df.columns if "-max" in c] #2021.08.11
  
  # print(pv_col) #['PV-max[0]', 'PV-max[1]', 'PV-max[2]', 'PV-max[3]']
  # sys.exit()
  
  _err=[]
  _col = [ f"{ec}({pc})" for pc in range(len(pv_col)) for ec in ["me","rmse","%mae","nrmse"]]
  for c in pv_col:
    df[c] = df[c].apply(lambda x: np.nan if x<0 else x)
  
  df.index.name = "time"
  # df.to_csv(f"{ESTIMATE}/{mm}_{radname}.csv",index=False)
  # df.to_csv(f"{ESTIMATE}/{mm}_{radname}_{cN}.csv")
  # print(df.head())
  # sys.exit()
  _mm = sorted(df["month"].unique().tolist())
  df = df[pv_col+["sum","month"]]
  df = df.dropna(subset=pv_col) #2021.11.19
  
  if 0: #debug
    df["ratio"] = (df["PV-max"] - df["sum"])/(df["sum"]+0.00001)
    df=df.sort_values("ratio", ascending=False)
    print(df.head())
    sys.exit()
    
  # e1,e2,e3 = get_err(df)
  err_hash={}
  
  for mm in _mm:
    tmp = df[df["month"]==mm]
    #----------------------------------
    tc="sum"
    _err=[]
    # _col = [ f"{ec}({pc})" for pc in range(len(pv_col)) for ec in ["me","rmse","%mae"]]
    _col = [ f"{ec}({pc})" for pc in range(len(pv_col)) for ec in ["me","rmse","%mae","nrmse"]]
    for pc in pv_col:
      # e1,e2,e3 = get_err(df,tc,pc)
      e1,e2,e3,e4 = get_err(tmp,tc,pc)
      _err += [e1,e2,e3,e4]
    
    err_hash[mm] = _err
    print(datetime.now(),"[end]", cate, mm, radname)
    #----------------------------------
  # print("2021.09.02 err start ...")
  df = pd.DataFrame(err_hash).T
  df.index.name = "mm"
  df.columns = _col
  df.to_csv(f"{ERR}/seido_mm_{cate}.csv")
  return

def calc_hh_seido(cate,seido,radname):
  path = f"{ESTIMATE}/PV_{seido}_{cate}_{radname}.csv"
  df = pd.read_csv(path)
  df["time"] = pd.to_datetime(df["time"])
  df = df.set_index("time")
  df["hhmm"] = df["hhmm"].apply(lambda x: str(x).zfill(4) )
  # print(df.head(10))
  # sys.exit()
  pv_col = [ c for c in df.columns if "-max" in c] #2021.08.11
  
  # print(pv_col) #['PV-max[0]', 'PV-max[1]', 'PV-max[2]', 'PV-max[3]']
  # sys.exit()
  
  _err=[]
  _col = [ f"{ec}({pc})" for pc in range(len(pv_col)) for ec in ["me","rmse","%mae","nrmse"]]
  for c in pv_col:
    df[c] = df[c].apply(lambda x: np.nan if x<0 else x)
  
  # df.index.name = "time"
  # df.to_csv(f"{ESTIMATE}/{mm}_{radname}.csv",index=False)
  # df.to_csv(f"{ESTIMATE}/{mm}_{radname}_{cN}.csv")
  # print(df.head())
  # sys.exit()
  _hh = sorted(df["hhmm"].unique().tolist())
  
  df = df[pv_col+["sum","hhmm"]]
  df = df.dropna(subset=pv_col) #2021.11.19
  
  if 0: #debug
    df["ratio"] = (df["PV-max"] - df["sum"])/(df["sum"]+0.00001)
    df=df.sort_values("ratio", ascending=False)
    print(df.head())
    sys.exit()
    
  # e1,e2,e3 = get_err(df)
  err_hash={}
  
  for hh in _hh:
    tmp = df[df["hhmm"]==hh]
    #----------------------------------
    tc="sum"
    _err=[]
    # _col = [ f"{ec}({pc})" for pc in range(len(pv_col)) for ec in ["me","rmse","%mae"]]
    _col = [ f"{ec}({pc})" for pc in range(len(pv_col)) for ec in ["me","rmse","%mae","nrmse"]]
    for pc in pv_col:
      # e1,e2,e3 = get_err(df,tc,pc)
      e1,e2,e3,e4 = get_err(tmp,tc,pc)
      _err += [e1,e2,e3,e4]
    
    err_hash[hh] = _err
    print(datetime.now(),"[end]", cate, hh, radname)
    #----------------------------------
  # print("2021.09.02 err start ...")
  df = pd.DataFrame(err_hash).T
  df.index.name = "hh"
  df.columns = _col
  df.to_csv(f"{ERR}/seido_hh_{cate}.csv")
  return

# ------------------------------------------------------------------------------

if __name__ == "__main__":
  
  if 1:
    for radname in ["8now0"]:
      for seido in ["train","test"]:
        #----
        # estimate(cate="all",seido=seido,radname = radname)
        # sys.exit()
        #----
        # calc_mm_seido(cate="all",seido=seido,radname = radname)
        calc_hh_seido(cate="all",seido=seido,radname = radname)
        sys.exit()
    sys.exit()