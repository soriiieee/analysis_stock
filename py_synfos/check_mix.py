# -*- coding: utf-8 -*-
"""[summary]
when   : 2021.04.13(作成したsynfos統合予測のec/synfosの欠測情報を確認するようなprogram)
update   : 2021.11.15

Returns:
    [type]: [description]
"""
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
#---------------------------------------------------
# sori -module
sys.path.append('/home/ysorimachi/tool')
from utils_plotly import plotly_1axis #(df,_col,html_path,title="sampe")
from tool_time import dtinc
from tool_matplot import set_japanese
#(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
import subprocess
import xml.etree.ElementTree as ET

power_dict ={
  0:"hkdn", #北海道
  1:"thdn", #東北
  2:"tden", #東京電力
  3:"hrdn",#北陸
  4:"cbdn",#中部
  5:"kndn",#関西
  6:"cgdn",#中国
  7:"yndn",#四国
  8:"kypv",#九州
  9:"okdn",#沖縄
}


imiss,rmiss = 9999,9999.
DAT_HOME="/work2/ysorimachi/synfos/mix"
# DAT_HOME="/work2/ysorimachi/synfos/data/SYNFOS-solar_ver2/recalc_hkrk/1" #2021.11.15
DAT_HOME="/work2/ysorimachi/synfos/data/SYNFOS-solar_ver2/recalc_okdn/2" #2021.11.25
#-LOG NONe-- 2021.11.15
LOG_NONE = f"/home/ysorimachi/work/synfos/log/log_nofile.dat" #output log
CSV_NONE = f"/home/ysorimachi/work/synfos/log/log_null.csv"
LOG_OUTLIER = f"/home/ysorimachi/work/synfos/log/log_outliers.dat"
CSV_OUTLIER = f"/home/ysorimachi/work/synfos/log/log_outliers.csv"

def list_DAY():
  _dd = sorted(os.listdir(DAT_HOME))
  return _dd

def load_data(path):
  # names= ["kdt","rrad","isyn","iecm","iecc","fecm","fecc"] #2021.11.15
  names= ["kdt","rrad","isyn","iecm","iecc","fecm","fecc","rCR0","rS0"] #2021.11.15
  df = pd.read_csv(path, delim_whitespace=True, header=None, names=names)
  df.loc[df["iecm"]==9999,"fecm"]=1
  df.loc[df["iecc"]==9999,"fecc"]=1
  df = df.iloc[1:,:]
  #null
  df = df.replace(9999,np.nan)
  #outliers
  for col in ["rrad","isyn","iecm","rCR0","rS0"]:
    df[col] = df[col].apply(lambda x: -9999 if (x != np.nan) and (x<0 or x>1300) else x)
  return df



def mk_CSV_None(code):
  """
  2021.04.13(Tue)
  統合予測で出力したdatファイルのうち、欠測記録を確認する
  """
  
  _dd = list_DAY()
  subprocess.run("rm -rf {}".format(LOG_NONE),shell=True)
  
  res={}
  for dd in tqdm(_dd):
    path = f"{DAT_HOME}/{dd}/{code}.dat"
    # print(path)
    # sys.exit()
    if os.path.exists(path):
      rad = load_data(path)
      # print(rad.head())
      # print(rad.columns)
      # sys.exit()
      # print(rad.isnull().sum().values[-6:])
      # sys.exit()
      res[dd] = list(rad.isnull().sum().values[-8:])
    else:
      with open(LOG_NONE,"+a") as f:
        text = f"NotFound {code} {dd}\n"
        f.write(text)
        
  res = pd.DataFrame(res).T
  res.columns = ["rrad","isyn","iecm","iecc","fecm","fecc","rCR0","rS0"]
  res["null_sum"] = res[["rrad","isyn","iecm","iecc","fecm","fecc","rCR0","rS0"]].sum(axis=1)
  res = res.sort_values("null_sum", ascending=False)
  res.to_csv(CSV_NONE)
  # 欠測の時刻についてnull_sum
  # list_ec_null = res.loc[res["null_sum"]>0,"Unnamed: 0"].values.tolist()
  # print(sorted(list_ec_null))
  # print(len(list_ec_null))
  # sys.exit()
  return

def null_check(code, isReturn=False,preMake=True):
  """
  2021.04.13(Tue)
  main() で出力したnullのログから、synfosとecの欠測数を表示する
  """
  # NULL_HOME="/home/ysorimachi/work/synfos/tmp/null"
  # path = f"{NULL_HOME}/{code}_null.csv"
  if preMake:
    mk_CSV_None(code) #pre make 2021.11.25
  df = pd.read_csv(CSV_NONE)
  
  N_FCT=len(df)
  N_DD=N_FCT//8
  
  null_ec = df[df["iecm"] !=0]
  null_syn = df[df["isyn"] !=0]
  
  if isReturn:
    return null_ec,null_syn
  percnt_ec = null_ec.shape[0]/len(df)
  percnt_syn = null_syn.shape[0]/len(df)
  # 
  print(null_ec.shape[0],"ECMWF null (",np.round(percnt_ec,5),"%) alltime->",N_FCT,"(",N_DD,"DAYS)")
  print(null_syn.shape[0],"SYNFOS-solar null (",np.round(percnt_syn,5),"%) alltime->", N_FCT,"(",N_DD,"DAYS)")
  return

  
def outliers_check(code):
  """
  2021.04.13(Tue)
  main() ：欠測記録の確認
  この場合、0~1300範囲外の記録を確認
  """
  _dd = list_DAY()
  df  =pd.DataFrame()
  # log_outliers = f"/home/ysorimachi/work/synfos/tmp/null/{code}_outliers.dat"
  subprocess.run("rm -rf {}".format(LOG_OUTLIER),shell=True)
  
  res={}
  for dd in tqdm(_dd):
    path = f"{DAT_HOME}/{dd}/{code}.dat"
    if os.path.exists(path):
      rad = load_data(path)
      
      _col = ["rrad","isyn","iecm"]
      n_rrad = rad[rad["rrad"]==-9999].shape[0]
      n_isyn = rad[rad["isyn"]==-9999].shape[0]
      n_iecm = rad[rad["iecm"]==-9999].shape[0]
      res[dd] = [n_rrad,n_isyn,n_iecm]
    else:
      with open(LOG_OUTLIER,"+a") as f:
        text = f"OutLiersCounts {dd}\n"
        f.write(text)
        
  res = pd.DataFrame(res).T
  res.columns = ["rrad","isyn","iecm"]
  res["sum"] = res[["rrad","isyn","iecm"]].sum(axis=1)
  res = res.sort_values("sum", ascending=False)
  res.to_csv(CSV_OUTLIER)
  return


def plot_rad_wave(code):
  NULL_HOME="/home/ysorimachi/work/synfos/tmp/null"
  _dd = list_DAY()[::8]
  ini_syn=_dd[0]
  n = len(_dd)
  
  # log_outliers = f"/home/ysorimachi/work/synfos/tmp/null/{code}_outliers.dat"
  # subprocess.run("rm -rf {}".format(log_outliers),shell=True)
  
  res={}
  _df = []
  col="rrad"
  for dd in tqdm(_dd):
    path = f"{DAT_HOME}/{dd}/{code}.dat"
    if os.path.exists(path):
      rad = load_data(path)
      rad["kdt"] = pd.to_datetime(rad["kdt"].astype(str))
      rad = rad.set_index("kdt")
      rad = rad.rename(columns={col : dd})
      _df.append(rad[dd])
  
  df = pd.concat(_df,axis=1)
  df.index.name="time"
  df = df.reset_index()

  #plotly--
  # html_path = f"{NULL_HOME}/html/{code}_init{ini_syn}_n{n}.html"
  # plotly_1axis(df,df.columns[1:],html_path,title=f"{code}_init{ini_syn}_n{n}")
  #png--
  f,ax = plt.subplots(figsize=(22,8))
  for c in df.columns[1:]:
    ax.plot(df[c])
  f.savefig(f"{NULL_HOME}/png/{code}_init{ini_syn}_n{n}.png", bbox_inches="tight")

def get_ec_ini(ini_j):
  if int(ini_j[:8]) < 20201210:
    ini_syn = dtinc(ini_j,4,-9)
    zz = ini_syn[8:10]
    
    if int(zz) != 21:
      return dtinc(ini_syn,3,-1)[:8]+"1200"
    else:
      return dtinc(ini_syn,3,0)[:8]+"1200"
  else:
    ini_syn = dtinc(ini_j,4,-9)
    ini_syn_yd = dtinc(ini_syn,3,-1)
    zz = ini_syn[8:10]
    if int(zz)==9 or int(zz)==12  or  int(zz)==15 or  int(zz)==18:
      return dtinc(ini_syn,3,-1)[:8]+"0000"
    elif int(zz)==0 or int(zz)==3  or  int(zz)==6:
      return dtinc(ini_syn_yd,3,-1)[:8]+"1200"
    else:
      return dtinc(ini_syn,3,0)[:8]+"1200"

def main2(code):
  ec_HOME="/mnt/ysorimachi/ecmwf_okada"
  ec,syn = null_check(code, isReturn=True, preMake=False)
  # print(ec.shape,syn.shape)
  # sys.exit()
  EC_CHECK_HOME="/home/ysorimachi/work/ecmwf/out/file_check"
  EC_CHECK2_HOME="/home/ysorimachi/work/ecmwf/out/file_check2"

  ec = ec.rename(columns={"Unnamed: 0" : "ini_syn"})
  ec["ini_j"] = ec["ini_syn"].apply(lambda x: dtinc(str(x),4,9))
  ec["ini_ec"] = ec["ini_j"].apply(lambda x: get_ec_ini(x))
  # print(ec["ini_ec"].unique())
  # print(syn.head())
  # sys.exit()
  # sys.exit()
  # _ini_ec = ec["ini_ec"].values.tolist()
  _ini_ec = ec["ini_ec"].unique()
  if 0:
    """ 実際にecが欠測な時間に、データが無いかの確認"""
    for ini_ec in _ini_ec:
      # ini_ec ="201912201200"
      path = f"{ec_HOME}/{ini_ec}/{code}_{ini_ec}_ecm.dat"
      if not os.path.exists(path):
        print(f"INI_EC({ini_ec}) is not Found..")
      else:
        df = pd.read_csv(path, delim_whitespace=True, header=None)
  if 1:
    """ データ取得時に、ecは取得できているのかの確認 """
    for ini_ec in _ini_ec:
      path= f"{EC_CHECK_HOME}/init_{ini_ec}_utc.dat"
      df = pd.read_csv(path, header=None, delim_whitespace=True)
      df[[3,4]].to_csv(f"{EC_CHECK2_HOME}/not_ec_ini{ini_ec}.csv", index=False,header=False)
  return 

def get_log_dd(log):
  df = pd.read_csv(log, header=None,skiprows=1,delim_whitespace=True)
  _dd = [ p for p in df[12].values.tolist() ]
  return _dd

def get_dir2(dd="20210403",name="recalc_hkrk",mtd=1):
  DHOME=f'/work2/ysorimachi/synfos/data/SYNFOS-solar_ver2/{name}/{mtd}'
  _hh = [ str(i*3).zfill(2) for i in range(8)]
  # _hh = np.roll(_hh,4)
  _tt = [ f"{dd}{hh}00" for hh in _hh ]
  _tt = [ dtinc(t,4,9) for t in _tt ]
  
  _dir = [f"{DHOME}/{ini_j}" for ini_j in _tt]
  return _dir
  



def plot_wave8(code,_dir):
  f,ax = plt.subplots(8,1,figsize=(10,18))
  ax = ax.flatten()
  set_japanese(fontsize=12) #軸を日本語ラベル化
  #-------------------------------
  for i,DIR in enumerate(_dir):
    path = f"{DIR}/{code}.dat"
    ini_u = path.split("/")[8]
    ini_j = dtinc(ini_u,4,9)
    # print(ini_u,ini_j)
    # sys.exit()
    df = load_data(path)
    df["kdt"] = pd.to_datetime(df["kdt"].astype(str))
    
    if int(ini_u) >= 202111090000:
      ecate="J2E"
    else:
      ecate = "D1E"
      
    _lbl=["統合予測","SYNFOS-solar",f"ECMWF({ecate})"]
    _col = ["rrad","isyn","iecm"]
    for j,(c,lbl) in enumerate(zip(_col,_lbl)):
      if j==0:
        lw=5
      else:
        lw=2
      ax[i].plot(df[c], label=lbl, lw=lw)
    #----------------
    ax[i].legend(loc="upper left")
    ax[i].set_ylim(0,1200)
    title = f"{ini_u}({ini_u[8:10]}Z)"
    ax[i].set_title(title)
  # save fig
  plt.subplots_adjust(wspace=0.4, hspace=0.4)
  f.savefig("./sample.png", bbox_inches="tight")
  return
  
  

    
if __name__=="__main__":
  # print(__name__) #__main__
  # print(__file__) #mix_check.py
  # print(__init__)
  # sys.exit()
  
  # code  ="okdn001"
  # code  = "tden001"
  code  = "hrdn011"
  if 0:
    """欠測記録の数を集計する"""
    # null_check(code)
    """外れ値の数を集計する"""
    # outliers_check(code)
    """ec欠測を表示する"""
    main2(code)
    
  if 1:
    """外れ値の数を集計する(描画等)"""
    # outliers_check(code)
    # _dir = get_log_dd(log="../log/loop_20211201.log")[:8]
    _dir = get_dir2(dd="20211111",name="recalc_hkrk",mtd=1)
    print(_dir)
    sys.exit()
    plot_wave8(code,_dir)
  
