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

sys.path.append('..')
try:
  from snow.utils_snow import load_snowdepth #(code="telm001")
except:
  from utils_snow import load_snowdepth #(code="telm001")
  
from mapbox import map_lonlat3 #(df,html_path,zoom=4,size_max=None)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

from utils_plotly import plotly_2axis #(df,col1,col2,html_path, title="sampe"):
from utils_plotly import plotly_1axis #(df,_col,html_path,title="sampe",vmax=1000)
TELEME="/work/ysorimachi/hokuriku/dat2/teleme/dataset" #30min_201912.csv
ESTIMATE2="/home/ysorimachi/work/hokuriku/out/teleme/pu/estimate2"

def load_PV(cate="train",mm=None,with_smame=False):
  if with_smame:
    path = f"{ESTIMATE2}/PV_{cate}_teleme_8now0_sm.csv"
  else:
    path = f"{ESTIMATE2}/PV_{cate}_teleme_8now0.csv"
  # print("load_PV->",path)
  df = pd.read_csv(path)
  df["time"] = pd.to_datetime(df["time"])
  df["mm"] = df["time"].apply(lambda x: x.strftime("%Y%m"))
  df = df.set_index("time")
  if mm:
    df = df[df["mm"]==mm]
  return df

def load_PV2(cate="train",dd=None,with_smame=False):
  if with_smame:
    path = f"{ESTIMATE2}/PV_{cate}_teleme_8now0_sm.csv"
  else:
    path = f"{ESTIMATE2}/PV_{cate}_teleme_8now0.csv"
    
  path = f"{ESTIMATE2}/PV_{cate}_teleme_8now0.csv"
  print("load_PV2->",path)
  df = pd.read_csv(path)
  df["time"] = pd.to_datetime(df["time"])
  sn = load_snowdepth("telm060").reset_index()
  sn = sn.rename(columns={"telm060": "snow" })
  
  df = df.merge(sn,on="time", how="left")
  # print(df.head())
  # sys.exit()
  df["time"] = pd.to_datetime(df["time"])
  # df["mm"] = df["time"].apply(lambda x: x.strftime("%Y%m"))
  df["dd"] = df["time"].apply(lambda x: x.strftime("%Y%m%d"))
  df = df.set_index("time")
  if dd:
    df = df[df["dd"]==dd]
  
  return df


def load_PV3(cate="train",mm=None,cate2="ind"):
  
  path = f"{ESTIMATE2}/PV_{cate}_teleme_8now0_{cate2}.csv"
  # print("load_PV->",path)
  df = pd.read_csv(path)
  df["time"] = pd.to_datetime(df["time"])
  df["mm"] = df["time"].apply(lambda x: x.strftime("%Y%m"))
  df["hh"] = df["time"].apply(lambda x: x.strftime("%H%M"))
  df = df.set_index("time")
  if mm:
    df = df[df["mm"]==mm]
  return df

def load_PV4(cate="train",dd=None,cate2="ind"):
  
  df0 = load_PV3(cate=cate,mm=None,cate2="sm")
  df1 = load_PV3(cate=cate,mm=None,cate2="ind")
  
  n=4 #1:ベンチマーク/4：月別時刻別
  # print(df.head())
  df = pd.concat([df0["sum"],df0[f"PV-max[1]"],df0[f"PV-max[{n}]"],df1[f"PV-max[{n}]"]],axis=1)
  # use_col = ['sum','PV-max[1]', f"PV-max[{n}]",f"PV-max[{n}]"]
  df.columns = ['sum','PV-max[1]', f"PV-max[{n}]",f"PV-max_ind[{n}]"]
  df = df.reset_index()
  df["time"] = pd.to_datetime(df["time"])
  df["dd"] = df["time"].apply(lambda x: x.strftime("%Y%m%d"))
  df["hh"] = df["time"].apply(lambda x: x.strftime("%H%M"))
  df = df.set_index("time")
  if dd:
    df = df[df["dd"]==dd]
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

def ts_TELEME_plotly(_mm):
  """[summary]

  Args:
      radname (str, optional): [description]. Defaults to "8now0".
      CSV (bool, optional): [description]. Defaults to False.

  Returns:
      [type]: [description]
  """
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
  #local function --------
  # _mm19 = loop_month(st="201904")[:12]
  # print(_mm)
  # print(_mm19)
  # sys.exit()
  
  # f,ax = plt.subplots(3,4,figsize=(4*5,3*4))
  # ax = ax.flatten()
  
  # OUTDIR="/home/ysorimachi/work/hokuriku/out/teleme/pu/png"
  OUTD="/home/ysorimachi/work/hokuriku/out/teleme/ts/plotly"
  for i, mm in enumerate(_mm):
  # for i,(mm,mm19)in enumerate(zip(_mm,_mm19)):
    # dataset
    # df = get_teleme_rad(mm,radname)
    # df = get_teleme_rad(mm,radname)
    if int(mm)>=202004:
      cate="test"
    else:
      cate="train"
      
    df = load_PV(cate=cate,mm=mm)
    df = df.reset_index()
    
    use_col = [ 'PV-max[0]', 'PV-max[1]',
       'PV-max[2]','PV-max[3]', 'PV-max[4]', 'PV-max_M1_SN0',
       'PV-max_M2_SN0', 'PV-max_M1_SN1', 'PV-max_M2_SN1']
    use_col = ['PV-max[1]','PV-max[4]', 'PV-max_M1_SN0','PV-max_M2_SN0']
    # print(df.head())
    # sys.exit()
    # names = [ 
    #          "手法1 : 2019/05(obs)",
    #          "手法2 : 2019/05(obs)+Hosei",
    #          "手法3 : 月[mm](8Now0)",
    #          "手法4 : 時刻[hh](8Now0)",
    #          "手法5 : 月[mm]-時刻[hh](8Now0)",
    #          "手法6 : Mesh(max)-Snow(なし)",
    #          "手法7 : Mesh(over)-Snow(なし)",
    #          "手法8 : Mesh(max)-Snow(あり)",
    #          "手法9 : Mesh(over)-Snow(あり)",
    # ]
    # df = df[use_col]
    # print(df.head())
    # print(df.columns)
    # df[use_col] /=100 
    vmax = df[use_col].describe().T["max"].max()//1000 * 1000 +1000
    # sys.exit()
    # # sys.exit()
    # # df = clensing(df,use_col = ["p.u","obs"],drop=False) #clensing
    # col1 = use_col
    # col2= ["sum"]
    _col = ["sum"] + use_col
    html_path = f"{OUTD}/{mm}.html"
    plotly_1axis(df,_col,html_path,title=f"{mm} : PV [TELEME]",vmax=vmax)
    # plotly_2axis(df,col1,col2,html_path, title=f"{mm} : PV [TELEME]",vmax=vmax)
    
    print(html_path)
    # sys.exit()
  return

def set_pv_col():
  use_col = [
    'PV-max[0]',
    'PV-max[1]',
    'PV-max[2]',
    'PV-max[3]',
    'PV-max[4]',
    'PV-max_M1_SN0',
    'PV-max_M2_SN0',
    'PV-max_M1_SN1',
    'PV-max_M2_SN1'
  ]
  names = [
    "2019/05(obs)",
    "2019/05(obs)+Hosei",
    "月[mm](8Now0)",
    "時刻[hh](8Now0)",
    "月[mm]-時刻[hh](8Now0)",
    "Mesh(max)-Snow(なし)",
    "Mesh(over)-Snow(なし)",
    "Mesh(max)-Snow(あり)",
    "Mesh(over)-Snow(あり)",
  ]
  names2 = [
    "手法1",
    "手法2",
    "手法3",
    "手法4",
    "手法5",
    "手法6",
    "手法7",
    "手法8",
    "手法9"
  ]
  # renames = {k:v for k,v in zip(use_col,names)}
  renames = {k:v for k,v in zip(use_col,names2)}
  
  # 非表示する場合のremove keys & elements ---------------
  # del renames['PV-max_M2_SN0']
  # del renames['PV-max_M2_SN1']
  # 非表示する場合の選択 ---------------
  return renames

def kaizen_ratio(df,c0="ベンチマーク",c1=None):
  if c1== None:
    sys.exit("please input c1 name !")
    
  e0 = rmse(df.iloc[:,0],df[c0]) #"bench mark!"
  e1 = rmse(df.iloc[:,0],df[c1])
  kaizen_Ratio = -1 * 100 * (e1 - e0)/e0 #[%] +/up -/down
  
  # e = np.round(e,1)
  kaizen_Ratio = np.round(kaizen_Ratio,2)
  return kaizen_Ratio

def ts_TELEME_dd_multi(_dd,plot="overfit"):
  OUTD="/home/ysorimachi/work/hokuriku/out/teleme/ts/pv_dd"
  if int(_dd[0][:6])>=202004:
    cate="test"
  else:
    cate="train"
  
  _df = []
  for dd in _dd:
    
    if plot != "setsubi":
      df = load_PV2(cate=cate,dd=dd)
      use_col = ["sum"] + [c for c in df.columns if c.startswith("PV-max")] + ["obs","snow"]
      df = df[use_col]
    else:
      df = load_PV4(cate=cate,dd=dd)
      df.loc[df["hh"]<="0530","PV-max_ind[4]"] = np.nan
      df.loc[df["hh"]>="1830","PV-max_ind[4]"] = np.nan
      
    _df.append(df)
  
  df = pd.concat(_df,axis=0)
  
    
    # print(df.head(50))
    # sys.exit()
  
  # print(df.head(30))
  # sys.exit()
  
  ylabel=r"PV出力[kW]"
  if 1:
    _col = ["sum"] + [ c for c in df.columns if "PV-max" in c]
    for c in _col:
      df[c] /= 1000 
    ylabel=r"PV出力[MW]"
    
  def calc_err(df,i):
    
    e0 = rmse(df.iloc[:,0],df.iloc[:,2]) #"bench mark!"
    e = rmse(df.iloc[:,0],df.iloc[:,i])
    kaizen = -1 * 100 * (e - e0)/e0 #[%] +/up -/down
    
    e = np.round(e,1)
    kaizen = np.round(kaizen,1)
    return e,kaizen

  def sub_bxplot(ax,df):
    bx = ax.twinx()
    bx.fill_between(x = np.arange(len(df)), y1= df["日射量"], y2=0,alpha=0.1,color="gray", label="日射量")
    bx.set_ylim(0,1)
    bx.set_ylabel(r"OBS-rad[W/$m^2$]")
    bx.legend(loc='upper right')
    return

  def sub_snow_bar(ax,df):
    bx = ax.twinx()
    # bx.fill_between(x = np.arange(len(df)), y1= df["日射量"], y2=0,alpha=0.3,color="yellow", label="日射量")
    bx.bar(np.arange(len(df)),df["積雪深"],alpha=0.1,color="blue", label="積雪深")
    # vmax = bx.get_ylim[1]*1.1
    # vmax = np.max(df["積雪深"])*4
    bx.set_ylim(0,200)
    bx.set_ylabel(r"Snowdepth[cm]")
    bx.legend(loc='upper right')
    return

  #------------------------------------------
  # print(df.head())
  if plot == "overfit":
    use_plot_col = [ 'sum', 'PV-max[1]','PV-max_M1_SN0', 'PV-max_M2_SN0']
    sub_col = ["obs"]
    renames,sub_name = [ '実発電量', 'ベンチマーク','設備量メッシュ(-)', '設備量メッシュ(過積載)'],["日射量"]
  elif plot == "snow":
    use_plot_col= [ 'sum', 'PV-max[1]','PV-max_M2_SN0', 'PV-max_M2_SN1']
    sub_col = ["snow"]
    renames,sub_name = [ '実発電量', 'ベンチマーク','設備量メッシュ(過積載)', '設備量メッシュ(過積載+積雪)'],["積雪深"]
  elif plot == "default":
    use_plot_col= [ 'sum', 'PV-max[1]','PV-max[3]', 'PV-max[4]']
    sub_col = []
    renames = ['実発電量', 'ベンチマーク','時刻別', '月別・時刻別']
    sub_name = []
  elif plot == "setsubi":
    use_plot_col= [ 'sum', 'PV-max[1]','PV-max[4]', 'PV-max_ind[4]']
    sub_col = []
    renames = ['実発電量', 'ベンチマーク','合算(teleme+スマメ全量)', '個別(teleme/スマメ全量)回帰']
    sub_name = []
  else:
    use_plot_col = "None"
  #------------------------------------------
  # renames = set_pv_col()
  # df = df.rename(columns=renames)
  df = df[use_plot_col + sub_col]
  df.columns = renames + sub_name
  
  df = df.replace(np.nan,0)
  _t = [ t.strftime("%H:%M") for t in df.index ] 
  dt = 12
  # print(df.head())
  # sys.exit()
  # sys.exit()
  rcParams['font.size'] = 24
  rcParams['ytick.labelsize'] = 24
  rcParams['xtick.labelsize'] = 24
  
  
  f,ax = plt.subplots(figsize=(32,10))
  #--------------------------
  if plot =="overfit":
    sub_bxplot(ax,df)
  if plot =="snow":
    sub_snow_bar(ax,df)
  
  if plot == "default" or plot == "setsubi":
    _c = df.columns[:]
  else:
    _c = df.columns[:-1]
  #---------------
  for i,c in enumerate(_c):
    if i == 0:
      lw=6
      # label=c
    else:
      lw=2
      # e,e_ratio = calc_err(df,i)
      # e_ratio = kaizen_ratio(df,c0="ベンチマーク",c1=c)
      # if e_ratio>0:
      #   flg="+"
      # else:
      #   flg=""
      # # label=f"{c}[RMSE= {e} /ratio= {flg}{e_ratio}%]"
      # label=f"{c}[改善率: {flg}{e_ratio}%]"
    ax.plot(np.arange(len(df)), df[c],lw=lw)
  ax.set_title(f"双方向端末データ-PV発電量(実況-予測){_dd[0]}~{_dd[-1]}")
  # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
  ax.legend(loc='upper left',fontsize=14)
  
  # if plot =="overfit":
  #   sub_bxplot(ax,df)
  # if plot =="snow":
  #   sub_snow_bar(ax,df)
  
  ax.set_ylabel(ylabel)
  
  #----------xaxis ---------
  ax.set_xlim(0,len(df)) #x軸の限界値を設定
  
  y0,y1 = 0,ax.get_ylim()[1] * 1.1
  ax.set_ylim(y0,y1) #x軸の限界値を設定
  st, ed = ax.get_xlim()
  ax.xaxis.set_ticks(np.arange(int(st), int(ed),dt))
  ax.set_xticklabels(_t[::dt],rotation=90)
  #----------xaxis ---------
  
  f.savefig(f"{OUTD}/pv_multi_{_dd[0]}.png", bbox_inches="tight")
  plt.close()
  print(datetime.now(),"[END]",f"{OUTD}/pv_{_dd[0]}.png")
  return

def points_TELEME_dd(dd):
  OUTD="/home/ysorimachi/work/hokuriku/out/teleme/ts/pv_dd"
  if int(dd[:6])>=202004:
    cate="test"
  else:
    cate="train"
  df = load_PV2(cate=cate,dd=dd)
  teleme_code = sorted([ c for c in df.columns if "telm" in c])
  df = df[teleme_code]
  # use_col = ["sum"] + [c for c in df.columns if c.startswith("PV-max")]
  # df = df[use_col]
  
  renames = set_pv_col()
  df = df.rename(columns=renames)
  df = df.rename(columns={"sum":"実発電量"})
  df = df.replace(np.nan,0)
  _t = [ t.strftime("%H:%M") for t in df.index ] 
  dt = 4
  # sys.exit()
  # print(df.shape)
  # sys.exit()
  
  f,ax = plt.subplots(6,10,figsize=(50,25))
  ax = ax.flatten()
  
  for i,c in enumerate(df.columns):
    ax[i].plot(np.arange(len(df)), df[c],lw=4)
    pv_max = teleme_max(c)
    title = f"{c}[{pv_max}]"
    ax[i].set_title(title)
    ax[i].tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
    ax[i].set_ylim(0,2500)
    
  for i in range(df.shape[1],60):
    ax[i].set_visible(False)
    
  # ax.set_title(f"双方向端末データ-PV発電量(実況-予測) -{dd}")
  # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
  # ax.set_ylabel(ylabel)
  
  #----------xaxis ---------
  # ax.set_xlim(0,len(df)) #x軸の限界値を設定
  # st, ed = ax.get_xlim()
  # ax.xaxis.set_ticks(np.arange(int(st), int(ed),dt))
  # ax.set_xticklabels(_t[::dt])
  
  #----------xaxis ---------
  plt.subplots_adjust(wspace=0.2, hspace=0.2)
  f.savefig(f"{OUTD}/point_{dd}.png", bbox_inches="tight")
  plt.close()
  print(datetime.now(),"[END]", f"{OUTD}/point_{dd}.png")
  return 

def code_TELEME_dd(dd,code="telm006"):
  OUTD="/home/ysorimachi/work/hokuriku/out/teleme/ts/pv_dd"
  if int(dd[:6])>=202004:
    cate="test"
  else:
    cate="train"
  df = load_PV2(cate=cate,dd=dd)
  # teleme_code = sorted([ c for c in df.columns if "telm" in c])
  # df = df[code]
  # print(df.head())
  # sys.exit()
  # use_col = ["sum"] + [c for c in df.columns if c.startswith("PV-max")]
  # df = df[use_col]
  max_pv = teleme_max(code=code,cate ="max")
  panel_pv = teleme_max(code=code,cate ="panel")
  # print(max_pv,panel_pv)
  # sys.exit()


  f,ax = plt.subplots(figsize=(18,10))
  # ax = ax.flatten()
  
# for i,c in enumerate(df.columns):
  ax.plot(np.arange(len(df)), df[code],lw=4)
  
  ax.axhline(y=max_pv, lw=4,color="k", label=f"MAX-PV={max_pv}[kW]")
  ax.axhline(y=panel_pv, lw=4,color="r", label=f"PANEL-PV={panel_pv}[kW]")
  
  
  ax.set_title(f"双方向端末({code})-PV発電量(実況-予測) -{dd}")
  # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
  ax.legend()
  # ax.set_ylabel(ylabel)
  # ax.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
  ax.set_ylim(0,3200)
    
  
  #----------xaxis ---------
  _t = [ t.strftime("%H:%M") for t in df.index ] 
  dt = 4
  ax.set_xlim(0,len(df)) #x軸の限界値を設定
  st, ed = ax.get_xlim()
  ax.xaxis.set_ticks(np.arange(int(st), int(ed),dt))
  ax.set_xticklabels(_t[::dt])
  
  #----------xaxis ---------
  # plt.subplots_adjust(wspace=0.2, hspace=0.2)
  f.savefig(f"{OUTD}/code_{dd}_{code}.png", bbox_inches="tight")
  plt.close()
  print(datetime.now(),"[END]", code, dd)
  return 

def improvement_dd(cate="train",with_smame=False):
  df = load_PV(cate=cate,mm=None,with_smame=with_smame)
  # print(df.head())
  # print(df.columns)
  # sys.exit(
  t0 = 'sum'
  c0 = 'PV-max[1]'
  
  if with_smame:
    use_col = ['sum','PV-max[1]', 'PV-max[2]','PV-max[3]', 'PV-max[4]']
    _c1 = ['PV-max[2]','PV-max[3]', 'PV-max[4]']
  else:
    use_col = ['sum','PV-max[1]', 'PV-max[2]','PV-max[3]', 'PV-max[4]', 'PV-max_M1_SN0', 'PV-max_M2_SN0','PV-max_M1_SN1', 'PV-max_M2_SN1']
    _c1 = ['PV-max[2]','PV-max[3]', 'PV-max[4]', 'PV-max_M1_SN0', 'PV-max_M2_SN0','PV-max_M1_SN1', 'PV-max_M2_SN1']
  
  df = df[use_col]
  #----------------
  def clensing(df):
    for c in use_col:
      df[c] = df[c].apply(lambda x: isFloat(x))
    df = df.dropna()
    return df
  
  def err_calc(df):
    _e=[]
    for c in [c0] + _c1:
      e = rmse(df[t0],df[c])
      _e.append(e)
    return _e
  
  def ratio_calc(df):
    _r=[]
    e0 = rmse(df[t0],df[c0])
    for c in [c0] + _c1:
      e = rmse(df[t0],df[c])
      r = -1 * (e - e0) *100 /e0 
      _r.append(r)
    return _r
  #----------------
  df = clensing(df)
  # print(df.head())
  # sys.exit()
  df = df.reset_index()
  df["dd"] = df["time"].apply(lambda x: x.strftime("%Y%m%d"))
  df = df.set_index("time")
  err_hash,imp_hash ={},{}
  _dd  =sorted(list(df["dd"].unique()))
  for dd in tqdm(_dd):
    tmp = df[df["dd"] == dd]
    _e = err_calc(tmp)
    _r = ratio_calc(tmp)
    err_hash[dd] = _e
    imp_hash[dd] = _r
    
  df = pd.DataFrame(imp_hash).T
  df.columns = [c0] + _c1
  df.index.name = "mm"
  df.to_csv(f"/home/ysorimachi/work/hokuriku/out/teleme/pu/select_dd/improvement_{cate}.csv")
  return


def improvement_dd_setsubi(cate="train"):
  
  df0 = load_PV3(cate=cate,mm=None,cate2="sm")
  df1 = load_PV3(cate=cate,mm=None,cate2="ind")
  
  n=4 #1:ベンチマーク/4：月別時刻別
  # print(df.head())
  df = pd.concat([df0["sum"],df0[f"PV-max[1]"],df0[f"PV-max[{n}]"],df1[f"PV-max[{n}]"]],axis=1)
  # use_col = ['sum','PV-max[1]', f"PV-max[{n}]",f"PV-max[{n}]"]
  df.columns = ['sum','PV-max[1]', f"PV-max[{n}]",f"PV-max_ind[{n}]"]
  t0 = 'sum'
  c0 = 'PV-max[1]'
  
  use_col = ['sum','PV-max[1]', f"PV-max[{n}]",f"PV-max_ind[{n}]"]
  _c1 = [f"PV-max[{n}]",f"PV-max_ind[{n}]"]
  
  df = df[use_col]
  #----------------
  def clensing(df):
    for c in use_col:
      df[c] = df[c].apply(lambda x: isFloat(x))
    df = df.dropna()
    return df
  
  def err_calc(df):
    _e=[]
    for c in [c0] + _c1:
      e = rmse(df[t0],df[c])
      _e.append(e)
    return _e
  
  def ratio_calc(df):
    _r=[]
    e0 = rmse(df[t0],df[c0])
    for c in [c0] + _c1:
      e = rmse(df[t0],df[c])
      r = -1 * (e - e0) *100 /e0 
      _r.append(r)
    return _r
  #----------------
  df = clensing(df)
  # print(df.head())
  # sys.exit()
  df = df.reset_index()
  df["dd"] = df["time"].apply(lambda x: x.strftime("%Y%m%d"))
  df = df.set_index("time")
  err_hash,imp_hash ={},{}
  _dd  =sorted(list(df["dd"].unique()))
  for dd in tqdm(_dd):
    tmp = df[df["dd"] == dd]
    _e = err_calc(tmp)
    _r = ratio_calc(tmp)
    err_hash[dd] = _e
    imp_hash[dd] = _r
    
  df = pd.DataFrame(imp_hash).T
  df.columns = [c0] + _c1
  df.index.name = "mm"
  df.to_csv(f"/home/ysorimachi/work/hokuriku/out/teleme/pu/select_dd/improvement_{cate}_setsubi.csv")
  return 


def select_dd(cate="train",n=30,GOOD=True,c1 ='PV-max[4]'):
  path = f"/home/ysorimachi/work/hokuriku/out/teleme/pu/select_dd/improvement_{cate}.csv"
  df = pd.read_csv(path)
  
  if GOOD:
    df = df.sort_values(c1,ascending=False)
  else:
    df = df.sort_values(c1)
    
  _dd = df["mm"].astype(str).values[:n:3].tolist()
  return _dd

def select_dd_setubi(cate="train",n=30,GOOD=True,c1 ='PV-max[4]'):
  path = f"/home/ysorimachi/work/hokuriku/out/teleme/pu/select_dd/improvement_{cate}_setsubi.csv"
  df = pd.read_csv(path)
  # print(df.head())
  # sys.exit()
  
  if GOOD:
    df = df.sort_values(c1,ascending=False)
  else:
    df = df.sort_values(c1)
    
  _dd = df["mm"].astype(str).values[:n:3].tolist()
  return _dd


if __name__ == "__main__":
  if 0: #月ごとの閲覧用
    _mm = loop_month()[12:24]
    ts_TELEME_plotly(_mm)
    sys.exit()
    
  if 0:
    # improvement_dd(cate="test",with_smame=True)
    improvement_dd_setsubi(cate="test")
    # _dd = select_dd(cate="test",GOOD=True)
  
  if 1:
    _dd = [
      # "20200819", #over_pv
      # "20200820", #over_pv
      # "20200821", #over_pv
      # "20200425", #over_pv
      # "20200513", #over_pv
      # "20200514", #over_pv
      # "20200613", #over_pv x
      # "20201017", #over_pv x
      "20210109", #snow
      "20210110", #snow
      "20210111",  #snow
      "20210112", #snow
      "20210113", #snow
      "20210114"  #snow
      #  '20200731', #setsubi
      #  '20200809', #setsubi
    ]

    # _c1 = ['PV-max[2]','PV-max[3]', 'PV-max[4]', 'PV-max_M1_SN0', 'PV-max_M2_SN0','PV-max_M1_SN1', 'PV-max_M2_SN1']
    #-------------
    # _dd = select_dd(cate="test",GOOD=True,c1 ='PV-max[4]')[:3]
    plot_type = "snow"
    
    #-------------
    # _dd = select_dd_setubi(cate="test",GOOD=True,c1 ='PV-max_ind[4]')[:10]
    # plot_type = "setsubi"
    # print(_dd)
    # sys.exit()
    ts_TELEME_dd_multi(_dd,plot=plot_type)