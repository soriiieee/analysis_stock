# -*- coding: utf-8 -*-
# when   : 2021.05.24
# who : [sori-machi]
# what : 
# *日射量(地上/衛星)との相関関係の把握
# *冬季期間における、富山・福井の日射量影響の把握/冬季期間においては、積雪により、日射量が過小評価になりがちなことを、plotlyで確認。
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
from utils_plotly import plotly_2axis#(df,col1,col2,html_path, title="sampe"):
# outd ='/home/griduser/work/sori-py2/deep/out/0924_01'
# os.makedirs(outd, exist_ok=True)
# subprocess.run('rm -f *.png', cwd=outd, shell=True)


DHOME="/work/ysorimachi/hokuriku/snow_hosei/rad210524" #8now0/sfc
DAT_1MIN="/home/ysorimachi/work/hokuriku/dat/rad/111600_2sites"
TMP="/home/ysorimachi/work/hokuriku/tmp/tmp210524"
point_hash = {"cpnt15": "47607","cpnt18": "47616"}
name_hash = {"cpnt15": "TOYAMA","cpnt18": "FUKUI"}

#settig loop
def load_month(isWinter=False):
  if isWinter:
    return list([202012,202101,202102,202103])
  else:
    return list([202004,202005,202006,202007,202008,202009,202010,202011,202012,202101,202102,202103])

# setting data load
def load_obs1min(code,month, ave=False):
  """
  2021.5.24
  荒井さんが作成された1min(気象官署データ)のload
  """
  path = f"{DAT_1MIN}/{code}_2020.csv"
  df = pd.read_csv(path)
  df.columns = ["time","kJ","rad"]
  df["time"] = pd.to_datetime(df["time"])
  df["mm"] = df["time"].apply(lambda x: x.strftime("%Y%m"))

  df = df[df["mm"]==str(month)]
  df = df.drop(["mm","kJ"],axis=1)
  if ave:
    df["rad"] = df["rad"].rolling(ave).mean()
    df = df.iloc[::ave]
    if month == 202004:
      df["time"] = df["time"].apply(lambda x: x-timedelta(minutes=1))
  
  df["time"] = df["time"].apply(lambda x: x.strftime("%Y%m%d%H%M"))
  return df

def load_8now0(code,month, ave=False):
  """
  2021.5.24
  﨤町が作成した1min(気象官署データ)のload
  """
  names = ["time","JWA_CODE","AMEDAS_CODE","JMA_Name","rlon","rlat","rHt0new","0.01*sd","0.01*ralb","rrad" ,"rrad_snow"]
  path = f"{DHOME}/8now0/ofile_{month}_{code}.dat"
  df = pd.read_csv(path,delim_whitespace=True, header=None,names=names)
  use_col = ["time","rHt0new","rrad"]
  df = df[use_col]
  df.columns =["time","H0","8now0"]
  df["time"] = df["time"].astype(str).apply(lambda x: x[:12])
  return df
  

def mk_rad_dataset(code):
  """
  main：プログラム(2021.05.24)
  8now0と111600の1minのデータを結合したデータセットを作成した。
  out：f"{DHOME}/dataset/{code}_{month}.csv"
  """
  for month in load_month(isWinter=False):
    # month="202005"
    # print(month)
    obs = load_obs1min(code,month, ave=5)
    sat = load_8now0(code,month)
    df = obs.merge(sat,on="time",how="inner")
    df.to_csv(f"{DHOME}/dataset/{code}_{month}.csv", index=False)
    print("end",month,code)
  return 

def scatter_rad(code,n_min=30):
  """
  scatter :
  8now0とobs1minの5分値データの相関関係を比較する
  """
  #local function ---------------
  def clensing(df):
    for c in ["rad","H0","8now0"]:
      df[c] = df[c].apply(lambda x: np.nan if x< 0 or x>1400 else x)
    df = df.dropna(subset=["rad","H0","8now0"])
    return df
  
  def average(df,n_min=5):
    if n_min==5:
      return df
    if n_min==30:
      for c in ["rad","H0","8now0"]:
        df[c] = df[c].rolling(6).mean()
      df = df.iloc[::6]
      return df
      
  
  def scatter(ax,df):
    r2_score = r2(df["rad"],df["8now0"])
    r2_score = np.round(r2_score,2)
    ax.scatter(df["rad"],df["8now0"], label=f"R2:{r2_score}\nsample:{df.shape[0]}", s=2)
    ax.plot(np.arange(1000),np.arange(1000),color="k",lw=1)
    ax.set_xlabel("obs(111600)")
    ax.set_ylabel("sat(8Now0)")
    ax.set_ylim(0,1200)
    ax.set_xlim(0,1200)
    ax.legend(loc="upper left")
    return ax
  #-cml---------------
  f,ax = plt.subplots(3,4,figsize=(4*3,3*3))
  ax = ax.flatten()
  _e=[]
  for i,month in enumerate(load_month(isWinter=False)):
    df = pd.read_csv(f"{DHOME}/dataset/{code}_{month}.csv")
    
    df = average(df,n_min=n_min)
    df = clensing(df)
    
    _e.append(r2(df["rad"],df["8now0"]))
    ax[i] = scatter(ax[i],df)
    ax[i].set_title(f"{month}({name_hash[code]})")
  #
  plt.subplots_adjust(wspace=0.4, hspace=0.5)
  f.savefig(f"{TMP}/{code}_scatter_rad_nmin{n_min}.png", bbox_inches="tight")
  df = pd.DataFrame()
  # df["month"] = [ str(m)[:4]+"年"+str(m)[4:]+"月" for m in load_month(isWinter=False)]
  df["month"] = load_month(isWinter=False)
  df["r2"] = _e
  df.to_csv(f"{TMP}/{code}_r2_score.csv",index=False)
  return

def check_rad(code,month):
  """
  2021.05.24 
  個別地点、個別の月において、日射量と積雪について確認する
  """
  #local function ---------------
  def clensing(df):
    for c in ["rad","H0","8now0"]:
      df[c] = df[c].apply(lambda x: np.nan if x< 0 or x>1400 else x)
    df = df.dropna(subset=["rad","H0","8now0"])
    return df
  
  def load_sfc(scode,month):
    path = f"{DHOME}/sfc/sfc_10minh_{month}_{scode}.csv"
    df = pd.read_csv(path)
    df = conv_sfc(df, ave=False)
    df["time"] = df["time"].apply(lambda x:x.strftime("%Y%m%d%H%M"))
    df = df[["time","snowDepth"]]
    df = df.replace(9999,np.nan)
    return df
  
  df = pd.read_csv(f"{DHOME}/dataset/{code}_{month}.csv")
  df["time"] = df["time"].astype(str)
  # df = clensing(df)
  scode = point_hash[code]
  ame = load_sfc(scode,month)
  
  # print(df)
  df = df.merge(ame, on="time",how="inner")
  df["time"] = pd.to_datetime(df["time"])
  html_path = f"{TMP}/{code}_{month}_rad_SNOW.html"
  plotly_2axis(df,["rad","8now0"],["snowDepth"],html_path, title=f"{code}_{month}_rad_SNOW")
  print(df.head())
  sys.exit()


if __name__ == "__main__":
  #---------------------------------------
  # #all months -------------
  if 1:
    for code in ["cpnt15","cpnt18"]:
      if 0:
        mk_rad_dataset(code)
      if 1:
        for n_min in [5,30]:
          scatter_rad(code,n_min=n_min)

  #---------------------------------------
  # 個別 -------------
  if 0:
    code = "cpnt15"
    month = "202006"
    check_rad(code,month)
  