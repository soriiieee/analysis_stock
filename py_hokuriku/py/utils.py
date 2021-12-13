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
from getErrorValues import me,rmse,mae,r2 #(x,y)
from convSokuhouData import conv_sfc #(df, ave=minutes,hour)
#amedas relate 2020.02,04 making...
from tool_AMeDaS import code2name, name2code
from tool_110570 import get_110570,open_110570
from tool_100571 import get_100571,open_100571
#(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
import subprocess

SMAME="/work/ysorimachi/hokuriku/dat2/smame/dataset2"
SET3="/work/ysorimachi/hokuriku/dat2/smame/set3_ts"
SMAME_TBL="/home/ysorimachi/work/hokuriku/tbl/smame"

def isFloat(x):
  try:
    v = float(x)
    return v
  except:
    return 9999.


def clensing_col(df,_col):
  for c in _col:
    df[c] = df[c].apply(lambda x: isFloat(x))
  return df




def loop_month(st = "201904", ed="202104"):
  _t = pd.date_range(start = f"{st}300000",end = f"{ed}300000", freq="M")
  _t = [ t.strftime("%Y%m") for t in _t]
  return _t

def loop_day(st = "201904", ed="202104"):
  _dd = [30,31, 30, 31, 31, 30, 31, 30, 31, 31, 29, 31, 30, 31, 30, 31, 31 ,30, 31, 30, 31, 31, 28, 31]
  return _dd


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



def load_rad(month="201904",cate="obs", lag=30,only_mean=False):
  """
  8now0 / obs のデータをダウンロードしてくるようなもの
  2021.10.04 make up !
  """
  # print(f"start ... load rad {cate}",month)
  if cate=="obs":
    path = f"/work/ysorimachi/hokuriku/dat2/rad/{cate}/1min/{month}_1min.csv"
    df = pd.read_csv(path)
    
  if cate=="8now0":
    path = f"/work/ysorimachi/hokuriku/dat2/rad/8Now0/5min/{month}_sat.csv"
    df = pd.read_csv(path)
    df = df.replace(9999,np.nan)
    df = df.rename(columns={"unyo000":"unyo009"})
    use_col = ["time"] + sorted([ c for c in df.columns if "kans" in c or "unyo" in c])
    df = df[use_col]
  # use_col.remove("unyo016")
  if "unyo016" in df.columns:
    df = df.drop(["unyo016"],axis=1)
  df["time"] = pd.to_datetime(df["time"])
  df["dd"] = df["time"].apply(lambda x: x.strftime("%Y%m%d"))
  
  #----------------------------------------------------------------------
  """
  2021.07.26  日時毎に異常値が入り込んでいるので除外処理を行う必
  2021.10.04  rad_jogai で個別に規定
  001_日射量.pptを参照に変更している
  """
  #----------------------------------------------------------------------
  # print(df.isnull().sum().sum()) #2021.10.04
  df = rad_jogai(df,cate=cate,dd=None)
  # print(df.isnull().sum().sum())
  
  use_col = [ c for c in df.columns if "unyo" in c]
  df["mean"] = df[use_col].mean(axis=1)
  df["std"] = df[use_col].std(axis=1)
  
  # print(df.head())
  # df.to_csv("/home/ysorimachi/work/hokuriku/dat/rad/r2/rad/tmp_check2.csv")
  df = df.drop_duplicates(subset =["time"])
  df = df.reset_index(drop=True)
  
  if lag:
    if cate=="8now0":
      lag = 6
    for c in use_col + ["mean","std"]:
      df[c] = df[c].rolling(lag).mean()
    
    df = df.iloc[::lag,:]
  df = df.set_index("time")
  
  #-----------------------
  if cate=="8now0":
    # print("DEBUG") # 2021.09.14 --> 3/24  は8Now0をゼロにする
    if month== "202003":
      df.loc[df["dd"]=="20200324","mean"] = np.nan
    if month== "202103":
      df.loc[df["dd"]=="20210324","mean"] = np.nan
  #-----------------------
  # df.to_csv("/home/ysorimachi/work/hokuriku/dat/rad/r2/rad/tmp_check2.csv")
  # sys.exit()
  if only_mean:
    df = df[["dd","mean","std"]]
  return df
  
# def get_a_b_c(month="202104",cate="obs"):
#   mm=month[4:6]
#   if cate == "obs":
#     path = "/home/ysorimachi/work/hokuriku/tbl/teleme/coef/ligression_values_unyo.csv"
#     df = pd.read_csv(path)
#     df["mm"] = df["month"].astype(str).apply(lambda x: x[4:6])
#     df = df[["mm","a","b","c"]].set_index("mm").T
#     para = df.to_dict()
#     return para[mm]["a"],para[mm]["b"],para[mm]["c"]
#   else:
#     sys.exit("please input obs")

def rad_jogai(df,cate="obs",dd=None):
  if cate =="obs":
    df.loc[df["dd"]=="20190418","unyo001"] = np.nan
    df.loc[df["dd"]=="20190801","unyo001"] = np.nan
    df.loc[df["dd"]=="20190904","unyo001"] = np.nan
    df.loc[df["dd"]=="20190904","unyo002"] = np.nan
    df.loc[df["dd"]=="20210224","unyo002"] = np.nan
    df.loc[df["dd"]=="20210224","unyo002"] = np.nan
    df.loc[df["dd"]=="20190603","unyo003"] = np.nan
    df.loc[df["dd"]=="20190604","unyo003"] = np.nan
    df.loc[df["dd"]=="20190605","unyo003"] = np.nan
    df.loc[df["dd"]=="20190606","unyo003"] = np.nan
    df.loc[df["dd"]=="20190620","unyo003"] = np.nan
    df.loc[df["dd"]=="20190904","unyo003"] = np.nan
    df.loc[df["dd"]=="20190904","unyo004"] = np.nan
    df.loc[df["dd"]=="20190904","unyo005"] = np.nan
    df.loc[df["dd"]=="20190930","unyo007"] = np.nan
    df.loc[df["dd"]=="20200228","unyo007"] = np.nan
    df.loc[df["dd"]=="20191106","unyo009"] = np.nan
    df.loc[df["dd"]=="20191114","unyo009"] = np.nan
    df.loc[df["dd"]=="20200227","unyo009"] = np.nan
    df.loc[df["dd"]=="20191128","unyo010"] = np.nan
    df.loc[df["dd"]=="20200219","unyo011"] = np.nan
    df.loc[df["dd"]=="20200121","unyo012"] = np.nan
    df.loc[df["dd"]=="20200226","unyo012"] = np.nan
    df.loc[df["dd"]=="20190515","unyo013"] = np.nan
    df.loc[df["dd"]=="20190527","unyo013"] = np.nan
    df.loc[df["dd"]=="20190528","unyo013"] = np.nan
    df.loc[df["dd"]=="20190529","unyo013"] = np.nan
    df.loc[df["dd"]=="20190530","unyo013"] = np.nan
    df.loc[df["dd"]=="20190531","unyo013"] = np.nan
    df.loc[df["dd"]=="20200821","unyo015"] = np.nan
    df.loc[df["dd"]=="20201215","unyo015"] = np.nan
    df.loc[df["dd"]=="20201216","unyo015"] = np.nan
    df.loc[df["dd"]=="20190517","unyo017"] = np.nan
    df.loc[df["dd"]=="20200518","unyo018"] = np.nan
    
  if cate =="8now0":
    _col = [ c for c in df.columns if "unyo" in c]
    for c in _col:
      df.loc[df["dd"]=="20200324",c] = np.nan
      df.loc[df["dd"]=="20210324",c] = np.nan
  # 除外ルール  -2020年度最終報告書　p22- 
  if dd:
    df = df[df["dd"]==dd]
  return df


    
def list_smame():
  path = "/home/ysorimachi/work/hokuriku/tbl/smame/list_smame.csv"
  df = pd.read_csv(path)
  
  #重複checkするけどlistにはなかった
  if 0:
    df["duplicated"] = df.duplicated(subset=["code"])
    print(df["duplicated"].sum())
    sys.exit()
  return df

def isFloat(x):
  try:
    v = float(x)
    return v
  except:
    return np.nan

def get_max_PV(_code):
  tbl = list_smame()
  tbl = tbl.loc[tbl["code"].isin(_code),:]
  return tbl["max[kW]"].sum()


def ts_smame_data(cate="all"):
  
  tbl = list_smame()
  num={}
  #---------------------------
  def clensing(df,_col):
    for c in _col:
      df[c] = df[c].apply(lambda x: isFloat(x))
    return df

  def calc_time(x,dd):
    if x=="2400":
      yy,mm,dd = int(dd[:4]),int(dd[4:6]),int(dd[6:8])
      t = datetime(yy,mm,dd,0,0)
      v = t + timedelta(days=1)
    else:
      v = pd.to_datetime(f"{dd}{x}")
    return v
  #---------------------------
  # for path in tqdm(_path):
  _mm = loop_month()
  
  for mm in _mm:
    _path = sorted(glob.glob(f"{SMAME}/{cate}_{mm}*.csv"))
    _f = [os.path.basename(path) for path in _path]
    
    _df=[]
    for path in _path:
      dd = os.path.basename(path).split("_")[1][:8]
      df = pd.read_csv(path)
      df2 = df.set_index("code")
      df2 = clensing(df2,_col=df2.columns)
      # print(dd,df.isnull().sum().sort_values(ascending=False))
      # print(df[df["code"]=="Z138930011200"])
      # 556  Z138930011200  < NULL >  < NULL >  < NULL >  < NULL >  < NULL >  < NULL >  < NULL >  < NULL > 
      df2 = df2.dropna()
      # print(df2.isnull().sum().sum())
      n_code = df2.shape[0]
      max_pv = get_max_PV(df2.index)
      
      
      sum_pv = df2.sum(axis=0)
      df = sum_pv.reset_index()
      df.columns = ["time", "sum"]
      df["time"] = df["time"].apply(lambda x: calc_time(x, dd=dd))
      
      df["count"] = n_code
      df["max"] = max_pv
      
      _df.append(df)
    
    df=pd.concat(_df,axis=0)
    df.to_csv(f"{SET3}/{cate}_{mm}.csv", index=False)
    
    print(datetime.now(), "[END]", cate,mm)
  return


def acode2scode(acode="11016"):
  path = "/home/ysorimachi/work/make_SOKUHOU3/tbl/list_sokuhou.csv"
  
  df = pd.read_csv(path)
  df["code"] = df["code"].astype(str)
  df = df.set_index("code")
  dict1 = df["scode"].to_dict()
  
  if acode in list(dict1.keys()):
    return dict1[acode]
  else:
    return 99999


def get_snowdepth(month,scode):
  """
  2021.07.27 sorimachi making
  """
  local = "/work/ysorimachi/hokuriku/dat2/snow_sfc/origin"
  path = f"{local}/sfc_10minh_{month}_{scode}.csv"
  df = pd.read_csv(path)
  df = conv_sfc(df,ave=30)
  df = df.replace(9999,np.nan)
  df["snowDepth"] = df["snowDepth"].fillna(method="pad")
  return df[["time","snowDepth"]]

if __name__ == "__main__":
  
  if 0:
    """
    2021.07.21 一度きり作成
    スマメデータの時系列作成
    """
    # for cate in ["all","surplus"]:
    for cate in ["surplus"]:
      ts_smame_data(cate = cate)
      sys.exit()
  
  if 1:
    df = load_rad(month="202102",cate="obs", lag=30)
    
  if 0:
    scode= acode2scode(acode="84519")
    print(scode)
    sys.exit()

    
  
  
  