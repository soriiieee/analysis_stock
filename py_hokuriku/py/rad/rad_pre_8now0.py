# -*- coding: utf-8 -*-
# when   : 2021.07.15
# who : [sori-machi]
# what : 北陸技研の記録についてセットアップする(異常値等の確認)
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

from plot1m import plot1m, plot1m_ec#(df,_col,_mem_col=False,month=False,vmin=0,vmax=1000,title=False)
from plot1d import plot1d_ec #title=False)
sys.path.append("..")
from utils import *
from sklearn.linear_model import LinearRegression

# outd ='/home/griduser/work/sori-py2/deep/out/0924_01'
# os.makedirs(outd, exist_ok=True)
# subprocess.run('rm -f *.png', cwd=outd, shell=True)

RAD_DAT="/home/ysorimachi/work/hokuriku/dat/rad/re_get"
OUT_1MIN = "/work/ysorimachi/hokuriku/dat2/rad/obs/1min"

OUT_8NOW0="/work/ysorimachi/hokuriku/dat2/rad/8Now0"


##---------------------------
def load_rad(code,dd=None):
  ###
  # 2021.09.14 SORIMACHI ADD 2019年度ルールの適応
  #
  ###
  # sub routine
  DHOME="/work/ysorimachi/hokuriku/dat2/rad/per_code"
  path = f"{DHOME}/rad_{code}.csv"
  df = pd.read_csv(path)
  df["time"] = pd.to_datetime(df["time"])
  df["dd"] = df["time"].apply(lambda x: x.strftime("%Y%m%d"))
  # 除外ルール  -2020年度最終報告書　p22- 
  if code=="unyo001":
    df.loc[df["dd"]=="20190418","obs"] = np.nan
    df.loc[df["dd"]=="20190801","obs"] = np.nan
    df.loc[df["dd"]=="20190904","obs"] = np.nan
  elif code == "unyo002":
    df.loc[df["dd"]=="20190904","obs"] = np.nan
    df.loc[df["dd"]=="20210224","obs"] = np.nan
    df.loc[df["dd"]=="20210224","obs"] = np.nan
  elif code == "unyo003":
    df.loc[df["dd"]=="20190603","obs"] = np.nan
    df.loc[df["dd"]=="20190604","obs"] = np.nan
    df.loc[df["dd"]=="20190605","obs"] = np.nan
    df.loc[df["dd"]=="20190606","obs"] = np.nan
    df.loc[df["dd"]=="20190620","obs"] = np.nan
    df.loc[df["dd"]=="20190904","obs"] = np.nan
  elif code == "unyo004":
    df.loc[df["dd"]=="20190904","obs"] = np.nan
  elif code == "unyo005":
    df.loc[df["dd"]=="20190904","obs"] = np.nan
  elif code == "unyo007":
    df.loc[df["dd"]=="20190930","obs"] = np.nan
    df.loc[df["dd"]=="20200228","obs"] = np.nan
  elif code == "unyo009":
    df.loc[df["dd"]=="20191106","obs"] = np.nan
    df.loc[df["dd"]=="20191114","obs"] = np.nan
    df.loc[df["dd"]=="20200227","obs"] = np.nan
  elif code == "unyo010":
    df.loc[df["dd"]=="20191128","obs"] = np.nan
  elif code == "unyo011":
    df.loc[df["dd"]=="20200219","obs"] = np.nan
  elif code == "unyo012":
    df.loc[df["dd"]=="20200121","obs"] = np.nan
    df.loc[df["dd"]=="20200226","obs"] = np.nan
  elif code == "unyo013":
    df.loc[df["dd"]=="20190515","obs"] = np.nan
    df.loc[df["dd"]=="20190527","obs"] = np.nan
    df.loc[df["dd"]=="20190528","obs"] = np.nan
    df.loc[df["dd"]=="20190529","obs"] = np.nan
    df.loc[df["dd"]=="20190530","obs"] = np.nan
    df.loc[df["dd"]=="20190531","obs"] = np.nan
  elif code == "unyo015":
    df.loc[df["dd"]=="20200821","obs"] = np.nan
    df.loc[df["dd"]=="20201215","obs"] = np.nan
    df.loc[df["dd"]=="20201216","obs"] = np.nan
  elif code == "unyo017":
    df.loc[df["dd"]=="20190517","obs"] = np.nan
  elif code== "unyo018":
    df.loc[df["dd"]=="20200518","obs"] = np.nan
  else:
    pass
  # 除外ルール　->　8Now0　メンテナンス？
  df.loc[df["dd"]=="20200324","8Now0"] = np.nan
  df.loc[df["dd"]=="20210324","8Now0"] = np.nan
  # 除外ルール  -2020年度最終報告書　p22- 
  if dd:
    df = df[df["dd"]==dd]
  return df

def plot1(mem_col,_df):
  """
  2021.08.30 # 地点別の日別日射量を見つける
  2021.09.14 # 地点別の日別日射量を見つける
  """
  dd = "20200416"
  f,ax = plt.subplots(4,5,figsize=(24,15))
  ax = ax.flatten()
  OUTD="/home/ysorimachi/work/hokuriku/dat/rad/ts/obs_dd"
  for i,(code,df) in enumerate(zip(mem_col,_df)):
    df = df[df["dd"]==dd]
    
    for c in ["obs","8Now0"]:
      ax[i].plot(np.arange(len(df)),df[c])
    ax[i].set_xlabel("時刻")
    # ax[i].set_xticks(np.arange(len(df)))
    # ax[i].set_xticklabels()
    ax[i].set_ylabel("日射量[W/m2]")
    ax[i].set_ylim(0,1200)
    
    title = f"{code}({rad_code2name(code)})"
    ax[i].set_title(title)
    
    #時刻ラベルを表記する 2021.09.08
    step=12
    # _t = [ t.strftime("%H:%M") for t in df.index]
    _t = [ t.strftime("%H") for t in list(df["time"])]
    ax[i].set_xticks(np.arange(len(df)))
    ax[i].set_xlim(0,len(df))
    st, ed = ax[i].get_xlim()
    ax[i].xaxis.set_ticks(np.arange(int(st), int(ed),step))
    ax[i].set_xticklabels(_t[::step], fontsize=12)
    
  plt.subplots_adjust(wspace=0.4, hspace=0.5)
  f.savefig(f"{OUTD}/rad_{dd}.png", bbox_inches="tight")
  print(OUTD)
  return

def plot2(mem_col,_df):
  """
  2021.08.30 # 地点別の散布図を表示するプログラム
  """
  f,ax = plt.subplots(4,5,figsize=(25,20))
  ax = ax.flatten()
  fontsize=14
  plt.rcParams['xtick.labelsize'] = fontsize        # 目盛りのフォントサイズ
  plt.rcParams['ytick.labelsize'] = fontsize        # 目盛りのフォントサイズ
  # plt.rcParams['figure.subplot.wspace'] = 0.20 # 図が複数枚ある時の左右との余白
  # plt.rcParams['figure.subplot.hspace'] = 0.20 # 図が複数枚ある時の上下との余白
  plt.rcParams['font.size'] = fontsize
  # plt.rcParams['lines.linewidth'] = 5
  
  OUTD="/home/ysorimachi/work/hokuriku/dat/rad/ts/obs_dd"
  
  # df = _df[2]
  # df= df[(df["obs"]<100)&(df["8Now0"]>500)] #2019年の異常値処理を挟むべきか
  # print(df)
  # sys.exit()
  for i,(code,df) in enumerate(zip(mem_col,_df)):
    df = df.dropna()
    df= df[(df["obs"]>0.1)&(df["8Now0"]>0.1)] # 2021.09.08
    # print(df.head())
    # print(df.describe())
    # sys.exit()
    e3 = np.round(r2(df["obs"],df["8Now0"]),2)
    # for c in ["obs","8Now0"]:
    
    # ax[i].scatter(df["obs"],df["8Now0"], s=1,color="r")
    ax[i].scatter(df["obs"],df["8Now0"], s=1,color="r",label=f"R2={e3}")
    ax[i].legend(loc="upper left")
    ax[i].plot(np.arange(1500), np.arange(1500), lw=1, color="k")
    
    ax[i].set_xlabel("地上運用観測点[W/m2]")
    ax[i].set_ylabel("8Now0[W/m2]")
    ax[i].set_ylim(0,1200)
    ax[i].set_xlim(0,1200)
    
    title = f"{code}({rad_code2name(code)})"
    ax[i].set_title(title)
    
  plt.subplots_adjust(wspace=0.5, hspace=0.5)
  f.savefig(f"{OUTD}/scatter_seido.png", bbox_inches="tight")
  print(OUTD)
  return

def seido2(mem_col,_df):
  """
  2021.08.30 # 地点別のerrorデータを表示するcsv作成program
  """
  def clensing(df ,filter_hh=[6,18]):
    #initil setting -------
    df = df.replace(0,np.nan) #もし0だったら、除外
    for c in ["obs","8Now0"]:
      df[c] = df[c].apply(lambda x: np.nan if x>1267 else x)
      df[c] = df[c].apply(lambda x: np.nan if x<10 else x) #10未満も除外
    #initil setting -------
    
    if "month" in df.columns:
      df = df.drop(["month"],axis=1)
    if "dd" in df.columns:
      df = df.drop(["dd"],axis=1)
    
    df["hh"] = df["time"].apply(lambda x: int(x.hour))
    if filter_hh:
      st,ed = filter_hh
      df = df[(df["hh"]>=st)&(df["hh"]<=ed)]
    df = df.dropna()
    return df
  
  
  OUTD="/home/ysorimachi/work/hokuriku/dat/rad/ts/obs_dd"
  err={}

  for i,(code,df) in enumerate(zip(mem_col,_df)):
    df1 = clensing(df,filter_hh=[6,18])
    df2 = clensing(df,filter_hh=[9,15])
    
    e1 = me(df1["obs"],df1["8Now0"])
    e2 = rmse(df1["obs"],df1["8Now0"])
    e3 = r2(df1["obs"],df1["8Now0"])
    e4 = mape(df2["obs"],df2["8Now0"])
    
    err[code] = [e1,e2,e3,e4]
  
  df = pd.DataFrame(err).T
  df.columns = ["me","rmse","r2","mape"]
  
  df.to_csv(f"{OUTD}/r2_err_point.csv")
  return 

def plot_2rad_month(dd="20200416"):
  """
  Main program 
  
  2021.08.30 
  unyoよ8now0　が各地点でどの程度、差があるのかをcheckするプログラム
  
  """
  mem_col = ["kans001","kans002"] + ['unyo001', 'unyo002', 'unyo003', 'unyo004', 'unyo005','unyo006', 'unyo007', 'unyo008', 'unyo009', 'unyo010', 'unyo011','unyo012', 'unyo013', 'unyo014', 'unyo015','unyo017','unyo018'] + ["mean"]
  
  _df=[]
  # ---------- 
  # 2021.09.14 -異常値に関してcheckする-
  def debug_rad(df):
    df = df.dropna()
    df = df[(df["obs"]>0.1)&(df["8Now0"]>0.1)]
    # df["ratio"] = df["8Now0"]/df["obs"]
    # df = df.sort_values("ratio", ascending=False)
    # print(df[df["8Now0"]>200].head(50))
    # sys.exit()
    return df
  # ---------- 
    
  for i,code in enumerate(mem_col):
    # code = "unyo015"
    df = load_rad(code)
    # df = debug_rad(df) #clensing & debug .. 2021.09.14
    # df = df[(df["obs"]<10)&(df["8Now0"]>100)]
    # df = df[(df["obs"]<10)&(df["8Now0"]<100)]
    # print(df.head(20))
    # sys.exit()
    _df.append(df)
  
  #-plot1(各地の日別)------------------------------
  # plot1(mem_col,_df)
  
  #-plot2(散布図)------------------------------
  plot2(mem_col,_df)
  seido2(mem_col,_df)
  return 


def dat2csv(month="202104"):
  DIR=f"/work/ysorimachi/hokuriku/dat2/rad/8Now0/{month}"
  path_col =  glob.glob(f"{DIR}/ofile_{month}_cpnt0.dat")[0]
  df = pd.read_csv(path_col, header=None, delim_whitespace=True)
  names = ["time"] + df.iloc[0,:].values.tolist()
  
  _path = glob.glob(f"{DIR}/ofile_*.dat")
  _path.remove(f"{DIR}/ofile_{month}_cpnt0.dat")
  
  for path in _path:
    df = pd.read_csv(path, header=None, delim_whitespace=True, names=names)
    df["JWA_CODE"] = df["JWA_CODE"].astype(str)
    
    code0 = df["JWA_CODE"].values[0]
    
    if str(code0)[0] != "9":
      code = f"47{code0}"
      if code =="47607":
        code = "kans001"
      elif code == "47616":
        code = "kans002"
      else:
        pass
    else:
      code0 = code0.replace("9","0")
      code = f"unyo{code0}"
    
    df =df[["time","rrad"]]
    
    if code[:4] == "unyo" or code[:4]== "kans":
      df.to_csv(f"{DIR}/{code}.csv", index=False)
    else:
      pass
  print(datetime.now(), "[END]", month)
  return



def csv_8now0(month="202104"):
  _path = sorted(glob.glob(f"{OUT_8NOW0}/{month}/*.csv"))
  
  _df = []
  _t = pd.date_range(start=f"{month}010000",periods=31*48*6, freq="5T")
  _t = [ t.strftime("%Y%m%d%H%M00") for t in _t]

  for path in _path:
    code = os.path.basename(path)[:7]
    
    df2 = pd.DataFrame()
    df2["time"] = _t
    df = pd.read_csv(path)
    df["time"] = df["time"].astype(str)
    
    df2 = df2.merge(df, on="time", how="left")
    df2 = df2.replace(np.nan, 9999)
    df2.columns = ["time",code]
    df2 = df2.set_index("time")
    _df.append(df2)
  
  df = pd.concat(_df,axis=1)
  df = df.reset_index()
  df["time"] = pd.to_datetime(df["time"])
  df["month"] = df["time"].apply(lambda x: x.strftime("%Y%m"))
  df = df[df["month"]==month]
  df = df.drop(["month"],axis=1)
  df = df.drop_duplicates(subset=["time"]) #2021.11.05 needs(複数時間帯がerrorで入ることがある)
  
  df.to_csv(f"{OUT_8NOW0}/5min/{month}_sat.csv", index=False)
  # print(df.shape)
  print(datetime.now(), "[END]", month)
  return



if __name__ == "__main__":
  if 1:
    _month = loop_month(st = "202104", ed="202210")[:7]
    _month = ['202107', '202108', '202109']
    # print(_month)
    # sys.exit()
    for month in _month:
      dat2csv(month) #dat->csv
      csv_8now0(month) #8now0-format
  
  if 0:
    """ 2021.08.30 add"""
    plot_2rad_month(dd = "20200416")