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
from getErrorValues import me,rmse,mae,r2, mape # 2021.09.02
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
from utils_teleme import *
from utils_data import teleme_and_rad,load_mesh_PV,load_mesh_PV2
sys.path.append("..")
from utils import *
from snow.utils_snow import load_snowdepth ,snow_PV_rate #(code="telm001")

from mapbox import map_lonlat3 #(df,html_path,zoom=4,size_max=None)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from plot1m import plot1m#(df,_col,vmin=0,vmax=1000,month=None,step=None,figtype="plot",title=False):

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
import matplotlib.colors as mcolors
_color = [ mcolors.rgb2hex(plt.cm.get_cmap('tab10')(i)) for i in range(10)]
TELEME="/work/ysorimachi/hokuriku/dat2/teleme/dataset" #30min_201912.csv


def get_err_model(err_name):
  if err_name =="rmse":
    return rmse
  if err_name == "me":
    return me

ESTIMATE="/home/ysorimachi/work/hokuriku/out/teleme/pu/estimate2"
# ERR="/home/ysorimachi/work/hokuriku/out/teleme/pu/err"
ERR="/home/ysorimachi/work/hokuriku/out/teleme/pu/err2"

def get_seido_month(cate):
  if cate == "train":
    _mm = loop_month(st="201904")[:12]
  else:
    _mm = loop_month(st="201904")[12:24]
  return _mm

def cut_hh_time(df,st=10,ed=14):
  df["hh2"] = df["time"].apply(lambda x: x.hour)
  df = df[(df["hh2"]>=st)&(df["hh2"]<=ed)]
  df = df.drop(["hh2"],axis=1)
  return df


#--------------------
def calc_mm_hh_PV(df,col,h_shift=1,m_shift=1): #2021.10.03
  """""
  時刻別の推定値を計算するsubroutine　#2021.10.03
  #2021.11.19
  Args:
      df ([type]): [description]
      col ([type]): [description]

  Returns:
      [type]: [description]
  """
  df = df.reset_index()
  df["hh"] = df["time"].apply(lambda x: x.strftime("%H%M"))
  df["mm"] = df["time"].apply(lambda x: x.strftime("%Y%m"))
  _hh = sorted(df["hh"].unique().tolist())
  mm = df["mm"].values[0]

  df[col] = 0
  for hh in _hh:
    a,b,c = load_mm_hh_abc(mm,hh,cate=cate)
    # print(mm,hh,a,b,c)
    # sys.exit()
    if a !=9999:
      df.loc[(df["mm"]==mm)&(df["hh"]==hh),col] = df.loc[df["hh"]==hh,["obs","sum_max"]].apply(lambda x: (a*x[0]**2 + b*x[0] + c )* x[1],axis=1)
    else:
      df.loc[(df["mm"]==mm)&(df["hh"]==hh),col] = 9999.
  df = df.set_index("time")
  df = df.drop(["mm"],axis=1)
  return df

def calc_hh_PV(df,col): #2021.10.03
  """""
  時刻別の推定値を計算するsubroutine　#2021.10.03
  #2021.11.19
  Args:
      df ([type]): [description]
      col ([type]): [description]

  Returns:
      [type]: [description]
  """
  df = df.reset_index()
  df["hh"] = df["time"].apply(lambda x: x.strftime("%H%M"))
  
  _hh = sorted(df["hh"].unique().tolist())
  
  df[col] = 0
  for hh in _hh:
    a,b,c = load_hh_abc(hh)
    if a !=9999:
      df.loc[df["hh"]==hh,col] = df.loc[df["hh"]==hh,["obs","sum_max"]].apply(lambda x: (a*x[0]**2 + b*x[0] + c )* x[1],axis=1)
    else:
      df.loc[df["hh"]==hh,col] = 9999.
  df = df.set_index("time")
  return df

def calc_mm_PV(df,mm):
  # 昨年度の踏襲も行う　-->　------ 21.09.20
  for i in range(3):
    # i=2
    if i==0:
      a,b,c = get_a_b_c("201905",cate="obs")
      hosei_ratio=1.
    if i==1:
      a,b,c = get_a_b_c("201905",cate="obs")
      hosei_ratio = get_hosei(mm,"T1") #21.12.07
    if i==2:
      a,b,c = get_a_b_c(mm,cate="8now0")
      hosei_ratio = 1.
      
      # print(a,b,c)
      # sys.exit() 
    #estimate
    df["pu-calc"] = df["obs"].apply(lambda x: a*x*x + b*x + c)
    df[f"PV-max[{i}]"] = (df["sum_max"] * df["pu-calc"]) * hosei_ratio
  return df


def calc_mesh(df,mm):
  # _point,cN = get_profile(p)
  
  _code = sorted(mk_teleme_table(ONLY_df=True)["code"].tolist())
  
  def month_flg(df):
    _code = [ f"{c}_FLG" for c in df.columns ]
    df = df.replace(np.nan,9999)
    _index = df.index
    for c in df.columns:
      df[c] = df[c].apply(lambda x: 0 if x==9999. else 1)
    df.columns = _code
    return df
  
  def mesh_pv_calc(df,drop=True,cname="sum",SNOW=1):
    #---------------------
    for c in _code:
      if SNOW:
        df[c] = df[c] * df[f"{c}_FLG"] * df[f"{c}_SNW"]
      else:
        df[c] = df[c] * df[f"{c}_FLG"] * 1 #補正無し
    #---------------------
    df = df[_code]
    df[cname] = df[_code].sum(axis=1)
    
    if drop:
      df = df[cname]
    return df
  
  def load_snow_hosei(df,mm,_code):
    sn = load_snowdepth(code=_code)
    _columns = [ c for c in df.columns if "telm" in c]
    _columns2 = [ f"{c}_SNW" for c in _columns] 
    sn = sn[_columns]
    
    _mm = np.unique([ t.strftime("%Y%m") for t in sn.index])
    # df["mm"] = df["mm"].apply(lambda x: x.strftime("%Y%m"))
    _index = df.index
    
    if not mm in _mm:
      
      flg = np.ones_like(df[_columns].values,dtype=np.float32)
      flg = pd.DataFrame(flg)
      flg.index = _index
      flg.columns = _columns2
      
    else:
      df = df.reset_index()
      sn = sn.reset_index()
      sn = sn.merge(df[["time","hh"]],on="time", how="inner")
      sn.set_index("time")
      sn = sn.drop(["hh"],axis=1)
      sn = sn[_columns]
      
      # flg = snow_PV_rate(sn.values,a=0.25,threshold=55)
      flg = snow_PV_rate(sn.values,a=0.25,threshold=20)
      flg = pd.DataFrame(flg)
      flg.index = _index
      flg.columns = _columns2
      
    return flg
  
  #----------------------
  for c in ['telm041', 'telm004', 'telm058', 'telm040']:
    _code.remove(c)
  # print(len(_code))
  # sys.exit()
  
  flg = month_flg(df[_code]) #月ごと
  # print(flg.shape)
  flg2 = load_snow_hosei(df,mm,_code)
  
  # print(df.head())
  # sys.exit()
  # for flg_s in [0,1]:
  
  _df=[]
  for flg_snow in [0,1]:
    for ver in [1,2]:
      # df2 = load_mesh_PV(mm,ver=ver) "月ごとの回帰式"
      df2 = load_mesh_PV2(mm,ver=ver) #月ごと時刻別の回帰
      # print(df2.head())
      # print(flg.head())
      # print(flg2.head())
      # sys.exit()
      df2 = pd.concat([df2,flg,flg2],axis=1)
      df2 = mesh_pv_calc(df2,drop=True,cname=f"PV-max_M{ver}_SN{flg_snow}",SNOW=flg_snow)
      _df.append(df2)
      # sys.exit()
      
  df2 = pd.concat(_df,axis=1)
  df = pd.concat([df,df2],axis=1)
  return df

#-----------------------------------------------
def get_err(df,tc,pc):
  """ 2021.09.02 """
  """ 2021.11.11 """
  df = df.reset_index()
  df["hh"] = df["time"].apply(lambda x: x.hour)
  # print(df.head())
  # print(df.shape)
  # sys.exit()
  d1 = df[(df["hh"]>=6)&(df["hh"]<=18)]
  d2 = df[(df["hh"]>=9)&(df["hh"]<=15)]
  
  e1 = me(d1[tc],d1[pc])
  # e1 = me(d1["PV-max"],d1["sum"])
  e2 = rmse(d1[tc],d1[pc])
  e3 = mape(d2[tc],d2[pc])
  
  e4 = rmse(d1[tc],d1[pc]) / np.mean(d1[tc])
  return e1,e2,e3,e4

def seido_mm(radname="8now0",p=90,cate="train",winter=False):
  """
  月別の誤差を表示するprogram
  init : 2021.08.11 start 
  update : 2021.09.02 
  update : 2021.09.20 
  update : 2021.10.01 #select 地点の確定と推定PVの確認 
  """
  # _point,cN = get_profile(p) 
    
  #local function --------
  _mm = get_seido_month(cate)
  if winter:
    _mm = _mm[-4:]
  # _mm19 = loop_month(st="201904")[:12]
  # print(_mm)
  # sys.exit()
  # print(_mm19)
  # sys.exit()
  
  f,ax = plt.subplots(3,4,figsize=(4*5,3*4))
  ax = ax.flatten()
  
  OUTDIR="/home/ysorimachi/work/hokuriku/out/teleme/pu/png"
  err_hash ={}
  # for i,(mm,mm19)in enumerate(zip(_mm,_mm19)):
  
  # err_hash = []
  _df_aa=[]
  for i ,mm in enumerate(_mm):
    
    # mm="202001"
    df = teleme_and_rad(radname=radname, mm=mm)
    # N_use=df["N"].values[10]
    df = clensing_col(df,_col = ["p.u","obs"]) #clensing
    # continue
    # sys.exit()
    # df1 = get_teleme_rad(mm19,radname)
    # df1 = clensing(df1,use_col = ["p.u","obs"],drop=True) #clensing
    
    #---------------------------------------------------------
    # ---------------------------------------
    # calc PV --
    df = calc_mm_PV(df,mm)
    df = calc_hh_PV(df,"PV-max[3]")
    df = calc_mm_hh_PV(df,"PV-max[4]")
    df = calc_mesh(df,mm)
    
    df = df.replace(9999,np.nan)
    _df_aa.append(df)
    
    # ---------------------------------------

    # pv_col = [ c for c in df.columns if "-max" in c or "-panel" in c]
    pv_col = [ c for c in df.columns if "-max" in c] #2021.08.11
    
    # print(pv_col) #['PV-max[0]', 'PV-max[1]', 'PV-max[2]', 'PV-max[3]']
    # sys.exit()
    
    _err=[]
    for c in pv_col:
      df[c] = df[c].apply(lambda x: np.nan if x<0 else x)
    
    df.index.name = "time"
    # df.to_csv(f"{ESTIMATE}/{mm}_{radname}.csv",index=False)
    # df.to_csv(f"{ESTIMATE}/{mm}_{radname}_{cN}.csv")
    # print(df.head())
    # sys.exit()
    df = df[pv_col+["sum"]]
    df = df.dropna(subset=pv_col) #2021.11.19
    
    # print(df.head())
    # print(df.describe())
    # sys.exit()
    
    if 0: #debug
      df["ratio"] = (df["PV-max"] - df["sum"])/(df["sum"]+0.00001)
      df=df.sort_values("ratio", ascending=False)
      print(df.head())
      sys.exit()
      
    # e1,e2,e3 = get_err(df)
    tc="sum"
    _err=[]
    
    # _col = [ f"{ec}({pc})" for pc in range(len(pv_col)) for ec in ["me","rmse","%mae"]]
    _col = [ f"{ec}({pc})" for pc in range(len(pv_col)) for ec in ["me","rmse","%mae","nrmse"]]
    for pc in pv_col:
      # e1,e2,e3 = get_err(df,tc,pc)
      e1,e2,e3,e4 = get_err(df,tc,pc)
      _err += [e1,e2,e3,e4]
    
    err_hash[mm] = _err
    # print("2021.09.02 err start ...")
    print(datetime.now(),"[end]", cate,mm, radname)
    # sys.exit()
  df = pd.DataFrame(err_hash).T
  # df.index= _mm
  df.index.name="month"
  df.columns = _col
  df.to_csv(f"{ERR}/seido_mm_{cate}_teleme_{radname}.csv")
  #---------------------------------------------------------
  
  df = pd.concat(_df_aa,axis=0)
  df.to_csv(f"{ESTIMATE}/PV_{cate}_teleme_{radname}.csv")
  print(datetime.now() , "[END]",f"{ERR}/seido_mm_{cate}_teleme_{radname}.csv")
  return

  
def seido_hh(radname="8now0",p=None,cate="train",winter=True,CI=False):
  """
  2021.09.20 : 時刻別評価
  2021.09.20 : 時刻別 -> selectで調整したデータ
  """
  cci=1 if CI else 0
  cwinter=1 if winter else 0
  
  # ERR="/home/ysorimachi/work/hokuriku/out/teleme/pu/err"
  # ESTIMATE="/home/ysorimachi/work/hokuriku/out/teleme/pu/estimate"
  # _df2 = [ pd.read_csv(f"{ESTIMATE}/{mm}_{radname}.csv") for mm in _mm2]
  
  # _point,cN = get_profile(p)
  _mm = get_seido_month(cate)
  if winter:
    _mm = _mm[-4:] 
  # print(_mm)
  # sys.exit()
  # _df = [ pd.read_csv(f"{ESTIMATE}/{mm}_{radname}.csv") for mm in _mm]
  _df =[]
  for i,mm in enumerate(_mm):
    df = teleme_and_rad(radname=radname,mm=mm)
    df = calc_mm_PV(df,mm)
    df = calc_mm_hh_PV(df,"PV-max[4]")
    df = calc_mesh(df,mm)
    _df.append(df)
  # sys.exit()
  
  
  df = pd.concat(_df,axis=0)
  #-------------
  df = calc_hh_PV(df,"PV-max[3]")
  #-------------
  df = df.reset_index()
  df["time"] = pd.to_datetime(df["time"])
  df["hh"] = df["time"].apply(lambda x: x.strftime("%H:%M"))
  df["dd"] = df["time"].apply(lambda x: x.strftime("%Y%m%d"))
  
  if CI:
    _dd = get_kaisei_dd(cate=cate,ci_thresold=0.7,st_h=10,ed_h=14)
    df = df.loc[df["dd"].isin(_dd),:]
    df = cut_hh_time(df,st=10,ed=14)
  
  # print(df.head())
  # sys.exit()
  
  _hh = sorted(df["hh"].unique().tolist())
  
  err_hash = {}
  for hh in _hh:
    _err=[]
    tmp = df[df["hh"]==hh]
    
    #---------------
    # a,b,c = load_hh_abc(hh.replace(":",""))
    # hosei_ratio=1
    # tmp["pu-calc"] = tmp["obs"].apply(lambda x: a*x*x + b*x + c)
    # tmp[f"PV-max[3]"] = (tmp["sum_max"] * tmp["pu-calc"]) * hosei_ratio 
    
  # print(df.head())
  # sys.exit()
    pv_col = sorted([ c for c in tmp.columns if "-max" in c]) #2021.08.11
    # print(pv_col)
    # sys.exit()
    
    _col = [ f"{ec}({pc})" for pc in range(len(pv_col)) for ec in ["me","rmse","%mae","nrmse"]]     
    for pc in pv_col:
      # e1,e2,e3= get_err(tmp,"sum",pc) #2021.09.03
      e1,e2,e3,e4= get_err(tmp,"sum",pc) #2021.10.04
      _err +=[e1,e2,e3,e4]
    
    err_hash[hh] = _err
  
  df = pd.DataFrame(err_hash).T
  df.index.name="hh"
  df.columns = _col
  df = df.replace(9999,np.nan)
  df.to_csv(f"{ERR}/seido_hh_{cate}_teleme_{radname}_wi{cwinter}_CI{cci}.csv")
  print(datetime.now() , "[END]",f"{ERR}/seido_hh_{cate}_teleme_{radname}_wi{cwinter}_CI{cci}.csv")
  return

    
def cut_time(df,st,ed):
  _hh = [ t.hour for t in df.index]
  df["hh"] = _hh
  df = df[(df["hh"]>=st)&(df["hh"]<=ed)]
  # df = df.reset_index()
  df = df.drop(["hh"],axis=1)
  return df

def select_teleme(threshold=95):
  _df=[load_teleme(mm) for mm in loop_month()]
  df = pd.concat(_df,axis=0)
  df = cut_time(df,st=9,ed=18)
  
  nul_Df = pd.DataFrame()
  nul_Df["point"] = df.columns
  nul_Df["n_data"] = df.shape[0]
  # print(df.isnull().sum().values)
  # sys.exit()
  nul_Df["n_nan"] = df.isnull().sum().values
  nul_Df["n_OK"] = nul_Df["n_data"] - nul_Df["n_nan"]
  nul_Df["%Get"] = nul_Df["n_OK"]*100/ nul_Df["n_data"]
  nul_Df = nul_Df.sort_values("%Get",ascending=False)
  nul_Df.to_csv("/home/ysorimachi/work/hokuriku/out/teleme/pu/tmp/null_points.csv", index=False)
  nul_Df = nul_Df[nul_Df["%Get"]>=threshold]
  # print(nul_Df)
  # print(nul_Df.shape)
  _point = nul_Df["point"].values.tolist()
  return _point


def get_profile(p):
  if p == "all":
    # _point = sorted(select_teleme(threshold=0))
    cN="all"
  elif type(p) == int:
    # _point = sorted(select_teleme(threshold=p))
    cN=f"sel{p}%"
  else:
    N = int(p.replace("select",""))
    _point = sorted(sonzai_sort(N=N))
    print(_point)
    # _point = ['telm032', 'telm016', 'telm030', 'telm034', 'telm037'] #  #debug -- 21.10.01
    cN="sel_SORI"
  return _point,cN 


if __name__ == "__main__":
  
  # _point = get_profile(p="select10")
  if 1:
    radname="8now0"
    for cate in ["train","test"]:
    # for cate in ["test"]:
      # seido_mm(radname=radname,p="all",cate=cate,winter=False) #月別 21.11.19 mesh-PVも()
      for winter in [0,1]:
        for CI in [0,1]:
          seido_hh(radname=radname,p="all",cate=cate,winter= winter,CI=CI) # 時刻別 21.12.03
      # sys.exit()
  
  if 0:
    for radname in ["8now0"]:
      # for p in [None]:
      for p in ["select"]:
        plot_ts_mm(mm="202101",radname=radname,p=p)