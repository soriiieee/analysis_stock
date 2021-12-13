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

sys.path.append("..")
from utils import load_rad,loop_month,loop_day,clensing_col
from utils_teleme import select_point,load_mm_hh_abc
from smame.utils_data import laod_smame_with_rad #(cate,mm)

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

def get_pu(df):
  df2 = max_df(df,cate="max")
  df3 = max_df(df,cate="panel")
  use_col = [ c for c in df.columns if "telm" in c ]
  
  df["sum"] = df[use_col].sum(axis=1)
  df["sum_max"] = df2[use_col].sum(axis=1)
  df["sum_panel"] = df3[use_col].sum(axis=1)
  df["p.u"] = df["sum"]/df["sum_max"]
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



# def load_mm_hh_abc(mm,hh):
#   """
#   0916 作成ただし、0900-1500の時間帯のみ実施
#   """
#   OUT_HOME="/home/ysorimachi/work/hokuriku/out/teleme/pu_hh"
#   # tmp = par[(par["mm"]==str(mm))&(par["hh"])]
#   m_shift,h_shift=1,1
#   path = f"{OUT_HOME}/param/regParam_m{m_shift}_h{h_shift}.csv"
#   par = pd.read_csv(path)
  
#   if cate == "train":
#     tmp = par[(par["mm"]==int(mm))&(par["hh"]==int(hh))]
#   if cate == "test":
#     mm=str(mm)[4:]
#     par["mm2"] =par["mm"].apply(lambda x: str(x)[4:])
#     tmp = par[(par["mm2"]==mm)&(par["hh"]==int(hh))]
#   a = tmp["a"].values[0]
#   b = tmp["b"].values[0]
#   c = tmp["c"].values[0]
#   return a,b,c

def load_hh_abc(hh):
  """
  1003 作成ただ
  """
  OUT_HOME="/home/ysorimachi/work/hokuriku/out/teleme/pu_hh"
  # tmp = par[(par["mm"]==str(mm))&(par["hh"])]
  m_shift,h_shift=1,1
  path = f"{OUT_HOME}/param/regParam_hh_2019.csv"
  par = pd.read_csv(path)
  
  tmp = par[par["hh"]==int(hh)]
  a = tmp["a"].values[0]
  b = tmp["b"].values[0]
  c = tmp["c"].values[0]
  return a,b,c


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


def teleme_and_rad(radname="8now0",mm="202004", select="all"):
  """[summary]

  Args:
      radname (str, optional): [description]. Defaults to "8now0".
      mm (str, optional): [description]. Defaults to "202004".

  Returns:
      DataFrame [pands]: 地点と日射量が両方含まれた時系列データの生成を行う予定。
  """
  use_code = select_point(N=select) #全地点を選択したとしても、10%以上はデータがある地点を出す
  #teleme データのload
  df = load_teleme(month=mm,min=30)
  df = df[use_code] #select
  
  df = get_pu(df) #teleme地点の最大を取得する
  
  rad = load_rad(month=mm,cate=radname, lag=30)
  rad = rad["mean"]/1000 #W->Kw
  rad.name = "obs"
  df = pd.concat([df,rad],axis=1)
  
  _min = np.unique([ c.minute for c in df.index])
  # print(mm,df.shape,_min)
  return df

def teleme_mesh_PV_set(mm,dd):
  """[summary]
  55server　で作成されたmesshPVの値をまとめたデータセットを作成する
  Args:
      mm (str, optional): [description]. Defaults to "202004".
  Returns:
      [type]: [description]
  """
  MESH="/work/ysorimachi/hokuriku/dat2/teleme/mesh_pv"
  MESH2="/work/ysorimachi/hokuriku/dat2/teleme/mesh_pv2"
  def preprocess(df):
    df["time"] = pd.to_datetime(df["time"].astype(str))
    _t = pd.date_range(start=f"{mm}010005",freq="5T",periods=dd*24*12)
    df_time= pd.DataFrame()
    df_time["time"] = _t
    df = df_time.merge(df,on="time", how="left")
    df = df.set_index("time")
    return df
  #-----------------
  
  _path = sorted(list(glob.glob(f"{MESH}/{mm}/*.dat")))
  names = list(pd.read_csv(_path[0],delim_whitespace=True).columns)

  _rad,_pv=[],[]
  _code =[]
  
  for p in tqdm(_path[1:]):
    df = pd.read_csv(p,header=None,delim_whitespace=True,names=names)
    df = preprocess(df)
    code = df.dropna()["code"].values[0]
    _code.append(code)
    rad = df[["rrad","rCR0"]]
    rad.columns = [code,f"{code}_CR0"]
    _rad.append(rad)
    pv = df[["pv_value","pv_value2"]]
    pv.columns =[f"{code}_1",f"{code}_2"]
    _pv.append(pv)
  
  rad = pd.concat(_rad,axis=1)
  pv = pd.concat(_pv,axis=1)
  rad.to_csv(f"{MESH2}/rad_{mm}.csv")
  pv.to_csv(f"{MESH2}/pv_{mm}.csv")
  # print(datetime.now(), "[end]", mm,dd)
  return

def load_mesh_PV(mm,ver=1,is30=True):
  # mm = "202103"
  MESH2="/work/ysorimachi/hokuriku/dat2/teleme/mesh_pv2"
  path = f"{MESH2}/pv_{mm}.csv"
  df = pd.read_csv(path)
  df["time"] = pd.to_datetime(df["time"])
  df = df.set_index("time")
  _code = sorted(list(np.unique([ c.split("_")[0] for c in df.columns])))
  use_col = [ c for c in df.columns if c.split("_")[1] ==str(ver)]
  df = df[use_col]
  df.columns = _code
  
  #----------------------2021.11.19------
  df = df.reset_index() #
  # df["d"] = df.duplicated(subset=["time"])
  # print(df[df["d"]==1])
  df = df.drop_duplicates(subset="time")
  # print(df.duplicated(subset=["time"]).sum())
  # print(mm,df["dup"].sum())
  #---------------------------
  df = df.set_index("time")
  if is30:
    for c in df.columns:
      df[c] = df[c].rolling(6).mean()
    df = df.iloc[5::6,:]
  
  df = df.drop(['telm040', 'telm041', 'telm004', 'telm058'],axis=1)
  _min = np.unique([ c.minute for c in df.index])
  # print(mm,df.shape,_min)
#   print(df.head())
#                telm001  telm002  telm003  telm005  telm006  telm007  telm008  ...  telm054  telm055  telm056  telm057  telm059  telm060  telm061
# time                                                                                ...
# 2019-04-01 00:30:00      NaN      NaN      NaN      NaN      NaN      NaN      NaN  ...      NaN      NaN      NaN      NaN      NaN      NaN      NaN
#   print(df.shape) #(1440, 57)
  # sys.exit()
  # print(df.describe())
  # print(df.shape)
  print(df.head(48))
  return df

def pre_mesh_PV2(mm,ver=1,is30=True):
  """[summary]
    設備日射量から、回帰式のa,b,cを独自に利用して回帰式毎の補正係数を用いるやり方
    2021.11.29
    2021.12.03 #update 選択的過積載の導入
  Args:
      mm ([type]): [description]
      ver (int, optional): [description]. Defaults to 1.
      is30 (bool, optional): [description]. Defaults to True.
  """
  # mm = "202103"
  MESH2="/work/ysorimachi/hokuriku/dat2/teleme/mesh_pv2"
  path = f"{MESH2}/rad_{mm}.csv"
  df = pd.read_csv(path)
  df["time"] = pd.to_datetime(df["time"])
  df = df.set_index("time")
  _code = sorted(list(np.unique([ c.split("_")[0] for c in df.columns])))
  # use_col = [ c for c in df.columns if c.split("_")[1] ==str(ver)]
  df = df[_code]
    
  #----------------------2021.11.19------
  df = df.reset_index() #
  df = df.drop_duplicates(subset="time")
  df = df.set_index("time")
  # print(df.duplicated(subset=["time"]).sum())
  # print(mm,df["dup"].sum())
  #---------------------------

  if is30:
    for c in df.columns:
      df[c] = df[c].rolling(6).mean()
    df = df.iloc[5::6,:]
  df = df.drop(['telm040', 'telm041', 'telm004', 'telm058'],axis=1)

  use_code = df.columns
  
  df = df.reset_index()
  df["hh"] = df["time"].apply(lambda x: x.strftime("%H%M"))
  df["mm"] = df["time"].apply(lambda x: x.strftime("%Y%m"))
  _hh = sorted(df["hh"].unique().tolist())
  mm = df["mm"].values[0]
  
  # print(df.head())
  df[use_code] /= 1000 
  
  def calc_pu(x,a,b,c,pv_max,pv_panel):
    # print("pv_max", pv_max)
    # print("pv_panel", pv_panel)
    # sys.exit()
    if ver==1:
      v = (a*x**2 + b*x + c ) * pv_max
    else:
      # --選択的に導入する(1000kW以上の大規模PVのみ導入)--
      if pv_max >= 1990:
        v = (a*x**2 + b*x + c ) * pv_panel
      else:
        v = (a*x**2 + b*x + c ) * pv_max
        # --選択的に導入する(1000kW以上の大規模PVのみ導入)--
    #------------------
    if v >= pv_max:
      v = pv_max
    elif v <=0:
      v = 0
    else:
      pass
    return v
  
  for col in use_code:
    pv_max = teleme_max(code=col,cate ="max")
    pv_panel = teleme_max(code=col,cate ="panel")
    # print(col,pv_max,pv_panel)
    # sys.exit()
    for hh in _hh:
      if int(mm)>=202004:
        cate="test"
      else:
        cate="train"
      a,b,c = load_mm_hh_abc(mm,hh,cate=cate,with_smame=False)
      # print(mm,hh,a,b,c)
      # sys.exit()
      if a !=9999:
        df.loc[(df["mm"]==mm)&(df["hh"]==hh),col] = df.loc[(df["mm"]==mm)&(df["hh"]==hh),col].apply(lambda x: calc_pu(x,a,b,c,pv_max,pv_panel))
      else:
        df.loc[(df["mm"]==mm)&(df["hh"]==hh),col] = 9999.
  df = df.set_index("time")
  df = df.drop(["mm","hh"],axis=1)
  df = df.replace(9999,np.nan)
  # print(df.describe())
  # print(df.shape)
  # print(df.head(48))
  # sys.exit()
  outpath = f"{MESH2}/pv_{mm}_mmhh_v{ver}.csv"
  df.to_csv(outpath)
  print(datetime.now(), "[end]", mm, ver)
  return

def load_mesh_PV2(mm,ver=1,is30=True):
  # mm = "202103"
  MESH2="/work/ysorimachi/hokuriku/dat2/teleme/mesh_pv2"
  path = f"{MESH2}/pv_{mm}_mmhh_v{ver}.csv"
  df = pd.read_csv(path)
  df["time"] = pd.to_datetime(df["time"])
  df = df.set_index("time")
  # print(df.describe())
  return df



def reg_use_data(mm="201905",hh="1200",radname="8now0",select="all",cate2="T1"):
  
  """[月別-時刻別回帰式のパラメータ作成]
  2021.09.15 月別の回帰係数を実施してみる
  2021.12.10 　解析に使用したデータのみの抽出
  Returns:
    None
      [type]: [description]
  """
  # subroutine functions --------------
  def list_hh(st="0600",ed="1800"):
    _hh = pd.date_range(start=f"20210101{st}", end=f"20210101{ed}", freq="30T")
    _hh = [t.strftime("%H%M") for t in _hh]
    return _hh
  
  def set_data(use_mm, use_hh):
    # _df = [ teleme_and_rad(radname=radname,mm=mm, select=select) for mm in use_mm]
    _df=[]
    for mm in use_mm:
      if cate2 =="T1":
        teleme_col=["p.u","obs","sum","sum_max"]
        df = teleme_and_rad(radname=radname, mm=mm,select="all")[teleme_col]
        df = clensing_col(df,_col = teleme_col) #clensing
      elif cate2 == "Z2":
        teleme_col=["p.u","obs","sum","sum_max"]
        df = teleme_and_rad(radname=radname, mm=mm,select="all")[teleme_col]
        df = clensing_col(df,_col = teleme_col) #clensing
      #------------------------------------------------------
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
      
      elif cate2 =="Z1":
        smame_col = ["p.u_sm","rad","sum","max_pv"]
        df = laod_smame_with_rad(cate="all",mm=mm)[smame_col]
        df = clensing_col(df,_col = smame_col) #clensing
        df = df.rename(columns = {"max_pv": "sum_max"})
        df = df.rename(columns = {"p.u_sm": "p.u"})
        df = df.rename(columns = {"rad": "obs"})
        df["obs"] /=1000 #->kW
        df["sum_max"] /= 2
        df["p.u"] = df["sum"] / df["sum_max"]
        
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
  _hh = list_hh(st="0530",ed="1830")
  # mm="201905"
  m2 = list_shift(mm,_month,n=-1)
  m1 = list_shift(mm,_month,n=1)
  use_mm = [m1,mm,m2]
  h1 = list_shift(hh,list_hh(),n=1)
  h2 = list_shift(hh,list_hh(),n=-1)
  use_hh = [h1,hh,h2]
  
  df = set_data(use_mm, use_hh)
  return df

if __name__ == "__main__":
  # v = load_point(code="telm003", cate="panel")
  # df = teleme_and_rad(radname="8now0",mm="202004")
  #--------------------------
  # _mm = loop_month()[:12]
  _mm = loop_month()[:24]
  # _dd = loop_day()[:12]
  _dd = loop_day()[:24]
  for (mm,dd) in zip(_mm,_dd): #2021.11.19
    # teleme_mesh_PV_set(mm,dd)
    # ----------------------------------------
    # load_mesh_PV(mm=mm)
    for ver in [1,2]:
      pre_mesh_PV2(mm=mm,ver=ver)
      # sys.exit()
      # load_mesh_PV2(mm=mm,ver=ver)
    # ----------------------------------------
  sys.exit()
  #--------------------------
  # print(df.head())
  sys.exit()

  