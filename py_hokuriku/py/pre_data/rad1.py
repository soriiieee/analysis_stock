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
from utils import *
from sklearn.linear_model import LinearRegression

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
  DIR="/home/ysorimachi/work/hokuriku/dat/rad/obs_1min"
  if int(month)>= 202004:
    p1 = f"{DIR}/2020_47607.csv"
    p2 = f"{DIR}/2020_47616.csv"
  else:
    p1 = f"{DIR}/2019_47607.csv" #2021.08.02
    p2 = f"{DIR}/2019_47616.csv" #2021.08.02
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

def pre_convert_rad():
  """
  2021.07.15
  2021.07.29 官署2地点も追加する 
  北陸技研からもらったフォーマットデータを扱いやすいように、地点合算したデータセットに直すプログラム
  main function ....
  2021.08.02 設定で実施する
  """
  # _month = loop_month()[:12] #2019
  _month = loop_month()[12:] #2020
  
  df = list_point(cate="unyo")
  _code = df["code"].values.tolist()
  _name = df["name"].values.tolist()
  
  for month in _month:
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
    df.to_csv(f"{OUT_1MIN}/{month}_1min.csv")
    
    # print(df.head())
    # print(df.describe())
    print(datetime.now(),"[end]", month)
    # sys.exit()
  return


def list_point(cate="unyo"):
  path = f"{RAD_DAT}/list_rad_point.csv"
  df = pd.read_csv(path)
  if cate:
    df["flg"] = df["code"].apply(lambda x: x.startswith(cate))
  df = df[df["flg"]==True].reset_index(drop=True)
  return df

def check_rad_ts(month="202004"):
  """
  2021.07.15 #月ごとに表示するプログラム
  2021.08.30 #monthに入力する形式に変更
  """
  # _month = loop_month()[12:]
  TS_OUT="/home/ysorimachi/work/hokuriku/dat/rad/ts/obs"
  unyo_col = [ "unyo" + str(i).zfill(3) for i in range(1,18+1)]
  unyo_col = [ "unyo" + str(i).zfill(3) for i in range(1,18+1)] + ["kans001","kans002"]
  unyo_col.remove("unyo016") #三国メガを外す
  
  # month ------------------
  def ave_rad(df):
    for c in unyo_col:
      df[c] = df[c].rolling(30).mean()
    
    # print(df.head()
    # sys.exit()
    df = df.iloc[::30,:]
    return df

  # for month in _month:
  path = f"{OUT_1MIN}/{month}_1min.csv"
  df = pd.read_csv(path)
  df["ave"] = df[unyo_col].mean(axis=1)
  df = ave_rad(df)
  df = df.dropna().reset_index(drop=True)
  
  # f = plot1m(df,_col=df.columns[1:],vmin=0,vmax=1200,month=month,step=240,figtype="plot",title=False)
  f = plot1m_ec(df,_col=["ave"]+["kans001","kans002"],_mem_col=unyo_col,month=month,vmin=0,vmax=1200,title=False)
  f.savefig(f"{TS_OUT}/{month}.png",bbox_inches="tight")
  plt.close()
  print(datetime.now(),"[end]", month)
  print("out director is -> ", TS_OUT)
    # sys.exit()
  return

def check_rad_ts_day(list_days):
  TS_OUT_DD="/home/ysorimachi/work/hokuriku/dat/rad/ts/obs_dd"
  # mem_col = ['unyo001', 'unyo002', 'unyo003', 'unyo004', 'unyo005',
  #      'unyo006', 'unyo007', 'unyo008', 'unyo009', 'unyo010', 'unyo011',
  #      'unyo012', 'unyo013', 'unyo014', 'unyo015', 'unyo016', 'unyo017',
  #      'unyo018'] #2021.07.14
  
  # mem_col = ["kans001","kans002"]
  mem_col = ['unyo001', 'unyo002', 'unyo003', 'unyo004', 'unyo005','unyo006', 'unyo007', 'unyo008', 'unyo009', 'unyo010', 'unyo011','unyo012', 'unyo013', 'unyo014', 'unyo015', 'unyo016', 'unyo017','unyo018'] + ["kans001","kans002"]
  
  mem_col2 = [ f"{c}({rad_code2name(c)})" for c in mem_col]
  rename_hash = {k:v for k,v in zip(mem_col,mem_col2)}
  
  def select_df(df,dd):
    df["time"] = pd.to_datetime(df["time"])
    df["dd"] = df["time"].apply(lambda x: x.strftime("%Y%m%d"))
    tmp = df[df["dd"]==dd]
    return tmp
  
  def mk_area_ave(df,_col):
    df["count"] = df[_col].count(axis=1)
    df["ave"] = df[_col].mean(axis=1)
    df.loc[df["count"]<16,"ave"]= np.nan
    df = df.drop(["count"],axis=1)
    return df
  
  vmax=1500
  for dd in list_days:
    month = dd[:6]
    path = f"{OUT_1MIN}/{month}_1min.csv"
    
    # print(path)
    df = pd.read_csv(path)
    df = select_df(df,dd)
    df = mk_area_ave(df,_col=mem_col) #2021.08.30
    mem_col += ["ave"]
    # print(df.iloc[400:450,:])
    # sys.exit()
    # df = df.rename(columns = rename_hash)
    # if 'unyo016(三国M-三国)' in mem_col2:
      # mem_col2.remove('unyo016(三国M-三国)')
    if 'unyo016' in mem_col:
      mem_col.remove('unyo016')
    
    if 1:
      df["ave30"] = df["ave"].rolling(30).mean()
      df = df.iloc[::30,:].reset_index(drop=True)
    
    # print(np.unique(df["time"].apply(lambda x: x.minute)))
    # print(df.head())
    # sys.exit()
    # print(df.describe().T["max"].max())
    # print(mem_col2)
    # sys.exit()
    if 1:
      # f = plot1d_ec(df,_col=mem_col2,_mem_col=False,dd=dd,vmin=0,vmax=vmax,title=dd, step=240)
      # f = plot1d_ec(df,_col=["ave","ave30"],_mem_col=None,dd=dd,vmin=0,vmax=vmax,title=dd, step=240)
      f = plot1d_ec(df,_col=["ave30"],_mem_col=None,dd=dd,vmin=0,vmax=vmax,title=dd, step=6)
      f.savefig(f"{TS_OUT_DD}/unyo_{dd}_vmax{vmax}.png",bbox_inches="tight")
      plt.close()
    
    print("end", dd, "dout->", TS_OUT_DD)
    # print("end", dd)
  return


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
  if 0:
    # 官署結合の為仕切り直し/地上観測点も同時取り込み
    pre_convert_rad()

  if 1:
    # check_rad_ts(month="202004") #month
    # sys.exit()
    #
    list_days = ["20200416"]
    check_rad_ts_day(list_days) #month
  
  if 0:
    _code = list_rad_col(isMean=True)
    for code in _code[11:]:      
      # print(code)
      # sys.exit()
      # code = "unyo009"
      concat_2rad(code,isSave=True)
      n_dd = check_2rad(code) #確認 2021.08.03
      print(datetime.now(),"[END]", code,"n_dd->",n_dd)
      # sys.exit()
      
    