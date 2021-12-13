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
from getErrorValues import me,rmse,mae,mape,r2 #(x,y)
from convSokuhouData import conv_sfc #(df, ave=minutes,hour)
#amedas relate 2020.02,04 making...
from tool_AMeDaS import code2name, name2code
from tool_110570 import get_110570,open_110570
from tool_100571 import get_100571,open_100571
#(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
import subprocess
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

def set_err_model(err_name):
  if err_name == "me":
    return me
  elif err_name == "rmse":
    return rmse
  elif err_name == "mae":
    return mae
  elif err_name == "mape":
    return mape
  else:
    sys.exit("please me/rmse/mae/mape")

def loop_month(st = "201904", ed="202104"):
  _t = pd.date_range(start = f"{st}300000",end = f"{ed}300000", freq="M")
  _t = [ t.strftime("%Y%m") for t in _t]
  _t = _t[:-1]
  return _t


def mk_rad_table():
  path = f"/home/ysorimachi/work/hokuriku/dat/rad/re_get/list_rad_point2.csv"
  df = pd.read_csv(path)
  # .set_index("code")
  return df

def mk_teleme_table(ONLY_df=False):
  """
  2021.07.13 最終的に調整する()
  """
  # path = "/home/ysorimachi/work/hokuriku/dat/teleme/re_get/TM地点情報.xlsx"
  # df = pd.read_excel(path,sheet_name= "Sheet1",engine='openpyxl',skiprows=1)
  path = "/home/ysorimachi/work/hokuriku/tbl/teleme/teleme_details.csv"
  df = pd.read_csv(path)
  df = df.rename(columns = {
    'No':"no",
    '地点名':"name", 
    '最大電力':"max", 
    'パネル容量':"panel"
  })
  df["no"] = df["no"].astype(int)
  df["code"] = df["no"].apply(lambda x: "telm"+ str(x).zfill(3))
  _name = df["name"].values.tolist()
  _code = df["code"].values.tolist()
  rename_hash = { k:v for k,v in zip(_name, _code)}
  
  if ONLY_df:
    return df
  else:
    return df, rename_hash


def teleme_max(code=None,cate ="max"):
  df,rename_hash = mk_teleme_table()
  df = df.set_index("code")[cate]
  dict1 = df.to_dict()
  
  if code:
    return dict1[code]
  else:
    return dict1


# def load_rad(month="201904",cate="obs", lag=30):
#   path = f"/work/ysorimachi/hokuriku/dat2/rad/{cate}/1min/{month}_1min.csv"
#   df = pd.read_csv(path)
#   # use_col.remove("unyo016")
#   df = df.drop(["unyo016"],axis=1)
#   df["time"] = pd.to_datetime(df["time"])
#   use_col = [ c for c in df.columns if "unyo" in c]
  
#   df["mean"] = df[use_col].mean(axis=1)
  
#   if lag:
#     for c in use_col + ["mean"]:
#       df[c] = df[c].rolling(lag).mean()
    
#     df = df.iloc[::lag,:]
#   df = df.set_index("time")
#   return df
def get_hosei(month="202104",cate="T1"):
  targt_mm = month[4:6]
  path =f"../../tbl/teleme/hosei/hosei_{cate}.csv"
  df = pd.read_csv(path)
  df["mm"] = df["mm"].astype(str).apply(lambda x: x[4:6])
  df = df.set_index("mm")
  hash_v = df.to_dict()["ratio"]
  return hash_v[targt_mm]

def sonzai_sort(N=5):
  df = detail_teleme()
  _index = df.index
  _columns = df.columns
  data = df.values
  data = np.where(data<0.8, np.nan,data)
  df = pd.DataFrame(data,index=_index,columns = _columns)
  df = df.dropna()
  
  df["ave"] = df[_columns].mean(axis=1)
  df["min"] = df[_columns].min(axis=1)
  # df["min"] = df[_columns].min(axis=1)
  df = df.sort_values(["ave"],ascending=False)
  _top = df.index[:N]
  return _top

def select_point(N=5):
  """
  teleme 地点の欠測率のデータから、存在している比率の多い、上位N地点のcodeを取得する
  Args:
      N (int, optional): [description]..

  Returns:
      _code : [欠測率が少ないような、codeを返す]
  """
  #---逐次作成---
  # df = detail_teleme()
  #---作済---
  df = pd.read_csv("/home/ysorimachi/work/hokuriku/dat/teleme/null/count/heatmap.csv")
  df = df.set_index("code")
  
  _index = df.index
  _columns = df.columns
  # data = df.values
  # data = np.where(data<0.8, np.nan,data)
  # df = pd.DataFrame(data,index=_index,columns = _columns)
  # df = df.dropna()
  # print(df.head(20))
  # print(df.shape)
  # sys.exit()
  
  df["ave"] = df[_columns].mean(axis=1)
  df["min"] = df[_columns].min(axis=1)
  # df["min"] = df[_columns].min(axis=1)
  df = df.sort_values(["ave"],ascending=False)
  
  #seelect 2021.11.09
  # print(df.shape)
  df = df[(df["ave"]>0.1)] #　データが10%以上含まれているような地点の抽出
  # print(df.shape)
  # sys.exit()
  
  if N=="all":
    _top = df.index
  else:
    _top = df.index[:N]
  return _top

def detail_teleme(CSV=False):
  ""
  TELEME="/work/ysorimachi/hokuriku/dat2/teleme/dataset"
  _month = loop_month()
  _res = []
  
  # _point =['telm032','telm016', 'telm030', 'telm034', 'telm037', 'telm009', 'telm010', 'telm033','telm007', 'telm022', 'telm012', 'telm018']
  # print(_point)

  
  # sys.exit()
  for month in _month:
    path = f"{TELEME}/30min_{month}.csv"
    df = pd.read_csv(path)
    
    use_col = [ c for c in df.columns if "telm" in c ]
    # use_col = _point
    n = df.shape[0]
    res = (n - df.isnull().sum())/n
    res = res[use_col]
    res.name = month
    _res.append(res)
  
  df = pd.concat(_res,axis=1)
  df = df.round(2)
  df.index.name = "code"
  df.to_csv("/home/ysorimachi/work/hokuriku/dat/teleme/null/count/heatmap.csv")
  
  f,ax = plt.subplots(figsize=(25,12))
  sns.heatmap(df,annot=True,vmin=0,vmax=1,ax=ax)
  ax.set_title("テレメデータ存在率[0=欠測]")
  f.savefig("/home/ysorimachi/work/hokuriku/dat/teleme/null/count/heatmap.png", bbox_inches="tight")
  return df

def load_mm_hh_abc(mm,hh,cate="train", with_smame=False):
  """
  0916 作成ただし、0900-1500の時間帯のみ実施
  """
  OUT_HOME="/home/ysorimachi/work/hokuriku/out/teleme/pu_hh"
  # tmp = par[(par["mm"]==str(mm))&(par["hh"])]
  m_shift,h_shift=1,1
  if with_smame:
    path = f"{OUT_HOME}/param/regPara_m{m_shift}_h{h_shift}_sm.csv"
  else:
    path = f"{OUT_HOME}/param/regPara_m{m_shift}_h{h_shift}.csv"
  par = pd.read_csv(path)
  # print(par.head())
  # sys.exit()
  
  if cate == "train":
    tmp = par[(par["mm"]==int(mm))&(par["hh"]==int(hh))]
  if cate == "test":
    mm=str(mm)[4:]
    par["mm2"] =par["mm"].apply(lambda x: str(x)[4:])
    tmp = par[(par["mm2"]==mm)&(par["hh"]==int(hh))]
  
  if tmp.empty:
    a,b,c = 9999,9999,9999
  else:
    a = tmp["a"].values[0]
    b = tmp["b"].values[0]
    c = tmp["c"].values[0]
  return a,b,c


def load_mm_hh_abc2(mm,hh,cate="train",cate2="Z2"):
  """
  0916 作成ただし、0900-1500の時間帯のみ実施
  """
  OUT_HOME="/home/ysorimachi/work/hokuriku/out/teleme/pu_hh"
  # tmp = par[(par["mm"]==str(mm))&(par["hh"])]
  m_shift,h_shift=1,1
  if cate2=="Z2":
    path = f"{OUT_HOME}/param/regPara_m{m_shift}_h{h_shift}_sm.csv"
  elif cate2 == "T1":
    path = f"{OUT_HOME}/param/regPara_m{m_shift}_h{h_shift}.csv"
  elif cate2 == "Z1":
    radname = "8now0"
    # path = f"/home/ysorimachi/work/hokuriku/tbl/smame/coef/param_hh_{radname}.csv"
    path = f"/home/ysorimachi/work/hokuriku/tbl/smame/coef/param_m{m_shift}_h{h_shift}_{radname}.csv"
  else:
    sys.exit("Z1/T1/Z2 !")
  
  par = pd.read_csv(path)
  # print(par.head())
  # sys.exit()
  
  if cate == "train":
    tmp = par[(par["mm"]==int(mm))&(par["hh"]==int(hh))]
  if cate == "test":
    mm=str(mm)[4:]
    par["mm2"] =par["mm"].apply(lambda x: str(x)[4:])
    tmp = par[(par["mm2"]==mm)&(par["hh"]==int(hh))]
  
  if tmp.empty:
    a,b,c = 9999,9999,9999
  else:
    a = tmp["a"].values[0]
    b = tmp["b"].values[0]
    c = tmp["c"].values[0]
  return a,b,c


def load_hh_abc(hh,with_smame=False):
  """
  1003 作成ただ
  """
  OUT_HOME="/home/ysorimachi/work/hokuriku/out/teleme/pu_hh"
  # tmp = par[(par["mm"]==str(mm))&(par["hh"])]
  m_shift,h_shift=1,1
  if with_smame:
    path = f"{OUT_HOME}/param/regParam_hh_2019_sm.csv"
  else:
    path = f"{OUT_HOME}/param/regParam_hh_2019.csv"
  par = pd.read_csv(path)
  par["hh"] = par["hh"].apply(lambda x: str(x).zfill(4))
  tmp = par[par["hh"]==str(hh)]
  
  if tmp.empty:
    return 9999,9999,9999
  else:
    a = tmp["a"].values[0]
    b = tmp["b"].values[0]
    c = tmp["c"].values[0]
    return a,b,c
  

def load_hh_abc2(hh,cate2):
  """
  12/08 作成(全量スマメ対応の層別化対応版)
  """
  OUT_HOME="/home/ysorimachi/work/hokuriku/out/teleme/pu_hh"
  # tmp = par[(par["mm"]==str(mm))&(par["hh"])]
  if cate2=="Z2":
    path = f"{OUT_HOME}/param/regParam_hh_2019_sm.csv"
  elif cate2 == "T1":
    path = f"{OUT_HOME}/param/regParam_hh_2019.csv"
  elif cate2 == "Z1":
    radname = "8now0"
    path = f"/home/ysorimachi/work/hokuriku/tbl/smame/coef/param_hh_{radname}.csv"
  else:
    sys.exit("Z1/T1/Z2 !")
    
  par = pd.read_csv(path)
  if cate2=="Z1":
    par = par.rename(columns = {"hhmm": "hh"})
    
  par["hh"] = par["hh"].apply(lambda x: str(x).zfill(4))
  tmp = par[par["hh"]==str(hh)]
  
  if tmp.empty:
    return 9999,9999,9999
  else:
    a = tmp["a"].values[0]
    b = tmp["b"].values[0]
    c = tmp["c"].values[0]
    return a,b,c
  
  

def get_a_b_c(month="202104",cate="obs",with_smame=False):
  """[summary]
  月別対応版の指標取得

  Args:
      month (str, optional): [description]. Defaults to "202104".
      cate (str, optional): [description]. Defaults to "obs".
      with_smame (bool, optional): [description]. Defaults to False.

  Returns:
      [type]: [description]
  """
  mm=month[4:6]
  if cate == "obs":
    if with_smame:
      path = "/home/ysorimachi/work/hokuriku/tbl/teleme/coef/ligression_values_obs_sm.csv"
    else:
      path = "/home/ysorimachi/work/hokuriku/tbl/teleme/coef/ligression_values_unyo.csv"
  if cate == "8now0": #sorimachi making 2021.09.02
    if with_smame:
      path = "/home/ysorimachi/work/hokuriku/tbl/teleme/coef/ligression_values_8now0_sm.csv"
    else:
      path = "/home/ysorimachi/work/hokuriku/tbl/teleme/coef/ligression_values_8now0.csv"
  # if cate == "8now0_with_smame": #sorimachi making 2021.09.08
  #   path = "/home/ysorimachi/work/hokuriku/tbl/teleme/coef/ligression_values_8now0_with_smame.csv"
    
  df = pd.read_csv(path)
  df["mm"] = df["month"].astype(str).apply(lambda x: x[4:6])
  df = df[["mm","a","b","c"]].set_index("mm").T
  para = df.to_dict()
  return para[mm]["a"],para[mm]["b"],para[mm]["c"]

def select_outlier(df,_col, err_name="rmse"):
  err_calc = set_err_model(err_name)
  if "dd" not in df.columns:
    df["dd"] = df["time"].apply(lambda x: x.strftime("%Y%m%d"))
  
  _dd = sorted(df["dd"].unique().tolist())
  
  c0,c1 = _col #比較する2columns
  _e=[]
  for dd in _dd:
    tmp = df[df["dd"] ==dd]
    e = err_calc(tmp[c0],tmp[c1])
    _e.append(e)
  
  df = pd.DataFrame()
  df["dd"] = _dd
  df["err"] = _e
  df = df.sort_values("err",ascending=False)
  return df


def isFloat(x):
  try:
    return float(x)
  except:
    return None
  
def teleme_max_info(cate="all"):
  ""
  TELEME="/work/ysorimachi/hokuriku/dat2/teleme/dataset"
  _point = ['telm032','telm016', 'telm030', 'telm034', 'telm037', 'telm009', 'telm010', 'telm033','telm007', 'telm022', 'telm012', 'telm018']
  
  df = teleme_max()
  if cate =="all":
    _point = list(df.keys())
  else:
    _point = ['telm032','telm016', 'telm030', 'telm034', 'telm037', 'telm009', 'telm010', 'telm033','telm007', 'telm022', 'telm012', 'telm018']
  #-------------------
  s=0
  for p in _point:
    if df[p]>0:
      s += df[p]
  #-------------------
  print(cate, len(_point),s)
  # sys.exit()
  return 

def get_kaisei_dd(cate="train",ci_thresold=0.7,st_h=10,ed_h=14):
  h0  = pd.read_csv(f"/work/ysorimachi/hokuriku/dat2/rad/tmp/teleme_H0.csv")[["time","mean"]]
  h0["time"] = pd.to_datetime(h0["time"])
  RAD="/work/ysorimachi/hokuriku/dat2/rad/per_code"
  
  # -------------------------
  def cut_time(df):
    df  = df[(df["hh"]>=st_h)&(df["hh"]<=ed_h)]
    return df
  def clensing(df):
    df["time"] = pd.to_datetime(df["time"])
    df["dd"] = df["time"].apply(lambda x: x.strftime("%Y%m%d"))
    df["hh"] = df["time"].apply(lambda x: x.hour )
    return df
  
  def calc_ci(df):

    df = df.merge(h0,on="time",how="inner")
    return df
  # ------------------------
  if cate=="train":
    _mm = loop_month()[:12]
  else:
    _mm = loop_month()[12:24]
  
  _rad = [ pd.read_csv(f"{RAD}/rad_mean_{mm}.csv") for mm in _mm]
  rad = pd.concat(_rad,axis=0)
  rad = clensing(rad)
  rad = cut_time(rad)
  rad = calc_ci(rad)
  rad["ki"] = rad["8now0"]/rad["mean"]
  rad = rad.groupby("dd").agg({"ki": ["mean","std"]})
  rad.columns = ["ki_mean","ki_std"]
  rad = rad.sort_values("ki_mean",ascending=False)
  # -----------
  rad = rad[rad["ki_mean"]>ci_thresold]
  _dd = rad.reset_index()["dd"]
  return _dd


def premake_H0():
  _t = pd.date_range(start="201904010000",freq="30T", periods=3 * 365 * 24 * 2)
  # print(_t)
  # sys.exit()
  from tool_sun import sun_position_wrapper
  
  _code,_H0=[],[]
  for i,r in mk_teleme_table(ONLY_df=True).iterrows():
    
    df = pd.DataFrame()
    df["dti"] = _t
    df = df.set_index("dti")
    if r["max"] == None:
      df["H0"] = 9999.
    else:
      df["LAT"] = r["lat"]
      df["LON"] = r["lon"]
      df["ALT"] = 0
      df["ANG"] = 0
      df = sun_position_wrapper(df)
    # ---
    _code.append(r["code"])
    _H0.append(df["H0"])
  
  df = pd.concat(_H0,axis=1)
  df.columns = _code
  df.index.name = "time"
  df["mean"] = df.mean(axis=1)
  # print(df.head())
  # sys.exit()
  df.to_csv(f"/work/ysorimachi/hokuriku/dat2/rad/tmp/teleme_H0.csv")
  return 
    
  



if __name__ == "__main__":
  # v = load_point(code="telm003", cate="panel")
  # df = load_rad()
  # acode2scode()
  # print(df.head())
  # print(v)
  hosei = get_hosei(month="202104",cate="T1")
  
  # a,b,c = load_hh_abc("0900")
  # print(a,b,c)
  
  # detail_teleme()
  # for cate in [ "all" ,"select12"]:
    # teleme_max_info(cate = cate)
  
  # df = mk_teleme_table(ONLY_df=True)
  # df.to_csv("../../tbl/teleme/teleme_details2.csv", index=False)
  # print(df.head())
  # sys.exit()
  
  # select_point()
  # premake_H0()
  get_kaisei_dd()
  