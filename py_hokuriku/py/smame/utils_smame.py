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
from mapbox import map_lonlat3 #(df,html_path,zoom=4,size_max=None)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

from matplotlib import rcParams
rcParams['font.size'] = 15
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

SMAME_SET3="/work/ysorimachi/hokuriku/dat2/smame/set3_ts" # 30min_201912.csv
COEF="/home/ysorimachi/work/hokuriku/tbl/smame/coef"
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

def smame_max(code=None):
  out_path = "../../tbl/smame/list_smame_near_rad.csv"
  df = pd.read_csv(out_path)
  
  if type(code) ==list:
    df = df.loc[df["code"].isin(code)]
    max_pv = df["max[kW]"].sum()
    return max_pv
  
def smame_max_code():
  out_path = "../../tbl/smame/list_smame_near_rad.csv"
  df = pd.read_csv(out_path)
  df = df[["code","max[kW]"]].set_index("code")
  # df = df.iloc[:10,:]
  df = df.to_dict()['max[kW]']
  return df
  
  
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

#-----------------------------------------------------------------------------------
# 2021.11.30 setting params 
# -------------------------------

def get_mm_abc_Z(mm, radname="8now0"):
  path =f"{COEF}/param_mm_{radname}.csv"
  df = pd.read_csv(path)
  df["month"] = df["month"].apply(lambda x: str(x).zfill(2))
  tmp = df[df["month"]==mm]
  a,b,c = tmp["a"].values[0],tmp["b"].values[0],tmp["c"].values[0]
  return a,b,c

def get_hh_abc_Z(hh, radname="8now0"):
  path =f"{COEF}/param_hh_{radname}.csv"
  df = pd.read_csv(path)
  df["hhmm"] = df["hhmm"].apply(lambda x: str(x).zfill(4))
  
  tmp = df[df["hhmm"]==hh]
  if tmp.empty:
    return 9999,9999,9999
  else:
    a,b,c = tmp["a"].values[0],tmp["b"].values[0],tmp["c"].values[0]
    return a,b,c

def get_mm_hh_abc_Z(mm,hh, radname="8now0"):
  path =f"{COEF}/param_m1_h1_{radname}.csv"
  df = pd.read_csv(path)
  df["hh"] = df["hh"].apply(lambda x: str(x).zfill(4))
  df["mm"] = df["mm"].apply(lambda x: str(x)[4:6])
  
  tmp = df[(df["hh"]==hh)&(df["mm"]==mm)]
  if tmp.empty:
    return 9999,9999,9999
  else:
    a,b,c = tmp["a"].values[0],tmp["b"].values[0],tmp["c"].values[0]
    return a,b,c

def get_smame_hosei(cate="all", mm="04"):
  """[summary]
  # 2021.11.30
    現行は、検針値で行った共通の指標を利用する予定(telemeと共通)
  Args:
      cate (str, optional): [description]. Defaults to "all".
      mm (str, optional): [description]. Defaults to "04".
  Returns:
      [type]: [description] Hosei ratio convert from raw PV to improved PV
  """
  hash_v = {
    "04": 1.173,
    "05": 1.077,
    "06": 1.109,
    "07": 1.136,
    "08": 1.079,
    "09": 1.224,
    "10": 1.280,
    "11": 1.356,
    "12": 1.456,
    "01": 1.445,
    "02": 1.308,
    "03": 1.300
  }
  return hash_v[mm]

if __name__ == "__main__":
  # get_a_b_c_d("201904","all")
  # smame_max()
  # a,b,c = get_mm_abc_Z("04", radname="8now0")
  # a,b,c = get_hh_abc_Z("0800", radname="8now0")
  a,b,c = get_mm_hh_abc_Z("04","0900", radname="8now0")
  