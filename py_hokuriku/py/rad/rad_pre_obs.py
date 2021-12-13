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
from getErrorValues import me,rmse,mae,r2 #(x,y)
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
sys.path.append('..')
from utils import *
from sklearn.linear_model import LinearRegression
from clensing import rad_JOGAI

# outd ='/home/griduser/work/sori-py2/deep/out/0924_01'
# os.makedirs(outd, exist_ok=True)
# subprocess.run('rm -f *.png', cwd=outd, shell=True)

RAD_DAT="/home/ysorimachi/work/hokuriku/dat/rad/re_get"
OUT_1MIN = "/work/ysorimachi/hokuriku/dat2/rad/obs/1min"


def get_path(code,name,month):
  """
  2021.07.13
  事前にRAD_DAT="/home/ysorimachi/work/hokuriku/dat/rad/re_get"以下に月フォルダを作成して、
  技研様からのフォーマットの日射量csvを格納しておく
  """
  # name -----------
  if code =="unyo003" or code =="unyo006" or code =="unyo008" or code =="unyo016":
    if int(month)>=201907:
      name = name.split("(")[1].replace(")","")
    else:
      name = name.split("(")[0]
  else:
    pass
  # name -----------
  path = f"{RAD_DAT}/{month}/{name}{month}日射.csv"
  # try:
  #   df = pd.read_csv(path)
  # except:
  #   print(code,name,month, "is not reading ... ")
    # sys.exit()
  return path

def read_rad(path):
  df = pd.read_csv(path, header=None, names= ["time","hh","mi","obs","count"])
  df = df.dropna()
  
  def get_time(x):
    yy = int(x[0].split("/")[0])
    mm = int(x[0].split("/")[1])
    dd = int(x[0].split("/")[2])
    hh = int(x[1])
    mi = int(x[2])
    return datetime(yy,mm,dd,hh,mi)
  
  df["time"] = df[["time","hh","mi"]].apply(lambda x: get_time(x),axis=1)
  df = df.drop(["hh","mi","count"],axis=1)
  df = df.set_index("time")
  return df

def load_kans(month="202004"):
  # DIR="/home/ysorimachi/work/hokuriku/dat/rad/obs_1min"
  DIR="/home/tarai/14_rikuden/giken/dat/kansho_1min"
  if int(month)< 202004:
    p1 = f"{DIR}/toyama_20190401-20200401.csv"
    p2 = f"{DIR}/fukui_20190401-20200401.csv"
  elif int(month) >= 202004 and int(month)< 202104:
    p1 = f"{DIR}/toyama_20200401-20210401.csv"
    p2 = f"{DIR}/fukui_20200401-20210401.csv"
  else:
    p1 = f"{DIR}/toyama_20210401-20211101.csv"
    p2 = f"{DIR}/fukui_20210401-20211101.csv"
    # sys.exit("not file !")
  
  _df=[]
  _code = ["kans001","kans002"]
  for path,code in zip([p1,p2],_code):
    df = pd.read_csv(path)[["TIME","GHI_W/m2"]]
    df.columns =[ "time",code]
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    _df.append(df)
  df = pd.concat(_df,axis=1)
  df.index.name = "time"
  df =df.reset_index()
  df["month"] = df["time"].apply(lambda x: x.strftime("%Y%m"))
  df =df[df["month"]==month]
  df = df.drop(["month"],axis=1)
  df = df.set_index("time")
  return df

def pre_convert_rad(month="201904"):
  """
  2021.07.15
  2021.07.29 官署2地点も追加する 
  北陸技研からもらったフォーマットデータを扱いやすいように、地点合算したデータセットに直すプログラム
  main function ....
  2021.08.02 設定で実施する
  2021.11.02 設定で実施する
  """

  _code = list_point(cate="unyo")["code"].values.tolist()
  _name = list_point(cate="unyo")["name"].values.tolist()


  kans = load_kans(month=month)
  # kans=kans.dropna()
  _rad=[]
  for code ,name in zip(_code,_name):
    path = get_path(code,name,month)
    rad = read_rad(path)
    rad = rad.rename(columns = {"obs":code})
    _rad.append(rad)
  
  df = pd.concat(_rad,axis=1)
  df = pd.concat([kans,df],axis=1)
  
  # clensing 処理(2021.11.02)　--- 
  df = df.reset_index()
  df = rad_JOGAI(df,cate="obs") #filtering -syori 
  df = df.set_index("time")
  # clensing 処理(2021.11.02)　--- 
  # OUT_1MIN = "/work/ysorimachi/hokuriku/dat2/rad/obs/1min"
  df.to_csv(f"{OUT_1MIN}/{month}_1min.csv")
  print(datetime.now(),"[end]", month)
  return

def check_outliers(month):
  """
  異常値のcheckを行う(2021.11.02)
  """
  PNG_DIR="/home/ysorimachi/work/hokuriku/dat/rad/ts/obs_check"
  if 1:
    subprocess.run(f"rm -f {PNG_DIR}/*.png",shell=True)
  
  def load_rad(month):
    rad_path = f"{OUT_1MIN}/{month}_1min.csv"
    df = pd.read_csv(rad_path)
    df["time"] = pd.to_datetime(df["time"])
    return df
  #-----------------------------
  # check [MONTH] ---(目視確認する為に月別で表示していく)
  if 1:
    df = load_rad(month)
    _col = [ c for c in df.columns if "unyo" in c]
    for c in tqdm(_col):
      # f = plot1m_ec(df,_col=[c],_mem_col=False,month=month,vmin=0,vmax=1200,title=False)
      f = plot1m(df,_col=[c],vmin=0,vmax=1200,month=month,step=None,figtype="plot",title=False)
      f.savefig(f"{PNG_DIR}/{month}_{c}.png", bbox_inches="tight")
      plt.close()
  #---------------------------
  #-----------------------------
  # check [dd] ---(目視確認する為に月別で表示していく)
  dd = f"20210831"
  if 0:
    df = load_rad(dd[:6])
    _col = [ c for c in df.columns if "unyo" in c]
    f = plot1d_ec(df,_col=_col,_mem_col=False,dd=dd,vmin=0,vmax=1200,title=f"SOLAR-RADIATION[day={dd}]- (ALL)",step=60)
    f.savefig(f"{PNG_DIR}/DD_{dd}.png", bbox_inches="tight")
  #---------------------------
  return



def list_point(cate="unyo"):
  path = f"{RAD_DAT}/list_rad_point.csv"
  df = pd.read_csv(path)
  if cate:
    df["flg"] = df["code"].apply(lambda x: x.startswith(cate))
  df = df[df["flg"]==True].reset_index(drop=True)
  return df


#------------------------------------
def list_rad_col(isMean=False):
  _code = ['kans001','kans002']+ ['unyo001', 'unyo002', 'unyo003', 'unyo004', 'unyo005','unyo006', 'unyo007', 'unyo008', 'unyo009', 'unyo010', 'unyo011','unyo012', 'unyo013', 'unyo014', 'unyo015','unyo017','unyo018']
  if isMean:
    _code = _code + ["mean"]
  return _code

def concat_2rad(code,isSave=False):
  """
  地上観測記録/8Now0の官署記録を結合するプログラム
  2021.07.15 平均日射量の違いをみるプログラム
  2021.08.02 官署データ記録の取り込みも行っている
  """
  # mem_col = ['unyo001', 'unyo002', 'unyo003', 'unyo004', 'unyo005','unyo006', 'unyo007', 'unyo008', 'unyo009', 'unyo010', 'unyo011','unyo012', 'unyo013', 'unyo014', 'unyo015','unyo017','unyo018']
  mem_col = list_rad_col() #2021.08.02 富山と福井を導入
  # "obs"
  
  _df=[]
  # for month in tqdm(loop_month()):
  for month in loop_month():
    # month = "202004"
    # "obs"
    path = f"{OUT_1MIN}/{month}_1min.csv"
    # print(path)
    obs = pd.read_csv(path)
    # print(obs.head())
    # print(obs.describe())
    # sys.exit()
    if code =="mean":
      obs["mean"] = obs[mem_col].mean(axis=1)
    obs = obs[["time",code]]
    obs[code] = obs[code].rolling(5).mean()
    obs = obs.iloc[::5,:]
    obs = obs.set_index("time")
    # 8Now0
    # path = f"/home/ysorimachi/work/hokuriku/dat/rad/8now0/master/{month}_sat.csv" #2021.07.15
    path = f"/work/ysorimachi/hokuriku/dat2/rad/8Now0/5min/{month}_sat.csv" #2021.08.03
    now8 = pd.read_csv(path)
    if int(month)>= 202004:
      now8 = now8.rename(columns = {"unyo000":"unyo009"})
    # print(now8.head())
    # sys.exit()
    if code =="mean":
      now8["mean"] = now8[mem_col].mean(axis=1)
    
    #--------------------------------
    # print(now8.shape) #重複を防ぐ為にかませる必要あり #2021.08.02
    now8 = now8.drop_duplicates(subset=["time"])
    # print(now8.shape)
    #--------------------------------
    now8 = now8.set_index("time")[code]
    
    # print(obs.head())
    # print(now8.head())
    # sys.exit()
    try:
      rad = pd.concat([obs,now8],axis=1)
      rad.columns = ["obs","8Now0"]
      rad = rad.reset_index()
      rad = rad.replace(9999,np.nan)
      # print("month", month, "OK!")
    except:
      rad = pd.concat([now8,obs],axis=1)
      rad.columns = ["8Now0","obs"]
      rad = rad[["obs","8Now0"]]
      rad = rad.reset_index()
      rad = rad.replace(9999,np.nan)
    
    for c in ["obs","8Now0"]:
      rad[c] = rad[c].rolling(6).mean()
    rad = rad.iloc[::6,:]
    rad["time"] = pd.to_datetime(rad["time"])
    rad["mi"] = rad["time"].apply(lambda x: x.strftime("%M"))
    rad["month"] = rad["time"].apply(lambda x: x.strftime("%Y%m"))
    _df.append(rad)
    
  #   print(month)
  
  # sys.exit()
  df = pd.concat(_df,axis=0)
  out_d = "/work/ysorimachi/hokuriku/dat2/rad/per_code"
  if isSave:
    df.to_csv(f"{out_d}/rad_{code}.csv", index=False)
    print(out_d)
  return df

def check_2rad(code):
  out_d = "/work/ysorimachi/hokuriku/dat2/rad/per_code"
  path = f"{out_d}/rad_{code}.csv"
  df = pd.read_csv(path)
  df["time"] = pd.to_datetime(df["time"])
  df["dd"] = df["time"].apply(lambda x: x.strftime("%Y%m%d"))
  
  # df = df[df["month"] == int("202010")]
  # print(df.head())
  # print(df.groupby("dd")["time"].count().sort_values(ascending=False))
  n_dd = df.groupby("dd")["time"].count().unique()
  return n_dd
 

if __name__ =="__main__":
  if 1: ## 地上観測データについて異常値検知
    _month = loop_month(st = "202104", ed="202503")[:7]
    # _month = ['202104', '202105', '202106', '202107', '202108', '202109', '202110']
    _month = ['202109']
    # print(_month[24:])
    # sys.exit()
    for month in _month:
      # pre_convert_rad(month) #前処理用のprogram
      # sys.exit()
      check_outliers(month) #目視確認用の関数
      # sys.exit()
    
    