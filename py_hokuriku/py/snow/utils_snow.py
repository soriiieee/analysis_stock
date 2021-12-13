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
# from tqdm import tqdm
# import seaborn as sns
#---------------------------------------------------
# sori -module
sys.path.append('/home/ysorimachi/tool')
from getErrorValues import me,rmse,mae,r2 #(x,y)
# from convSokuhouData import conv_sfc #(df, ave=minutes,hour)
# from convSokuhouData import conv_sfc #(df, ave=minutes,hour)
from convAmedasData import conv_amd #(df, ave=minutes,hour)
#amedas relate 2020.02,04 making...
from tool_AMeDaS import code2name, name2code
from distance import calc_km #(lon_a,lat_a,lon_b,lat_b)
# from tool_110570 import get_110570,open_110570
# from tool_100571 import get_100571,open_100571
#(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
import subprocess

sys.path.append("..")
from utils import mk_rad_table


def load_data(path):
  df = pd.read_csv(path)
  if "time" in df.columns:
    if df["time"].dtypes == object:
      df["time"] = pd.to_datetime(df["time"])
  return df
  

def get_amecode(list_name):
  # for name in list_name:
  list_code = [ name2code(name) for name in list_name]
  return list_code

def set_name():
  list_name=["朝日","魚津","伏木","富山","砺波","氷見","猪谷","秋ケ島","珠洲","輪島","七尾","金沢","白山河内","加賀菅谷","福井","大野","今庄","敦賀","小浜","九頭竜","武生"]
  return list_name

def set_year(YY):
  #2019------------------
  if YY==2019:
    _month = ["201904","201905","201906","201907","201908","201909","201910","201911","201912","202001","202002","202003"]
    _day = [30,31,30,31,31,30,31,30,31,31,29,31]
  #2020------------------
  if YY==2020:
    _month = ["202004","202005","202006","202007","202008","202009"]
    _day = [30,31,30,31,31,30]
  return _month,_day
  
def mk_point_list(FTP=False):
  """
  必要なアメダス地点リストを作成して、bin/ftp.shで取得する　->/work/ysorimachi/hokuriku/dat/snow/amedas
  """
  list_name = set_name()
  list_code = get_amecode(list_name)
  
  
  with open("./list_snow.dat", "w") as f:
    for code, name in zip(list_code, list_name):
      if code != 'nan':
        text = f"{code} {name}\n"
        f.write(text)
  
  if FTP:
    # --> 作成した地点リストのアメダス月報を57号機から、->/work/ysorimachi/hokuriku/dat/snow/amedasへ
    com1 = "rm /work/ysorimachi/hokuriku/dat/snow/amedas/*.csv"
    com2 = "sh /home/ysorimachi/work/hokuriku/bin/ftp.sh"
    for com in [ com1,com2]:
      subprocess.run(com,shell=True)
  
  return
  
  
def mk_year_dataset():
  tbl = pd.read_csv("./list_snow.dat", delim_whitespace=True,header=None, names=["code","name"])
  _name = tbl["name"].values.tolist()
  _code = tbl["code"].values.tolist()
  _month, _day = set_year(2019)
  
  
  DHOME="/work/ysorimachi/hokuriku/dat/snow/amedas"
  OHOME="/work/ysorimachi/hokuriku/dat/snow/amedas2"

  for name , code in zip(_name,_code):
    
    _df = []
    for month , day in zip(_month,_day):
      path = f"{DHOME}/amd_10minh_{month}_{code}.csv"
      if os.path.exists(path):
        df = pd.read_csv(path)
        df = conv_amd(df,ave=30)
        _df.append(df)
    
    df = pd.concat(_df,axis=0)
    df.to_csv(f"{OHOME}/{code}.csv", index=False)
    print("end", code, name)
    
def ave_data(col):
  DHOME = f"/work/ysorimachi/hokuriku/dat/snow/amedas2"
  _path = glob.glob(f"{DHOME}/[0-9][0-9][0-9][0-9][0-9].csv")
  
  _df=[]  
  for path in _path:
    code = os.path.basename(path)[:5]
    df = load_data(path)
    df = df.set_index("time")
    df = df[col]
    df.name = code
    _df.append(df)
  
  df = pd.concat(_df,axis=1)
  df.to_csv(f"{DHOME}/ave_{col}.csv")
  return 

def clensing(df,col):
  if col=="snowDepth":
    use_col = df.iloc[:,1:].columns
    # print(use_col)
    # sys.exit()
    for col in use_col:
      df[col] = df[col].apply(lambda x: np.nan if x>1000 else x)
  else:
    pass
  return df


def load_ame_weather(col):
  # ["tenminPrecip","temp","tenminSunshineTime","snowDepth"]
  DHOME="/work/ysorimachi/hokuriku/dat/snow/amedas2"
  path = f"{DHOME}/ave_{col}.csv"
  df = load_data(path)
  df = clensing(df,col)
  return df

def load_teleme_position():
  path = "/home/ysorimachi/work/hokuriku/tbl/teleme/teleme_details.csv"
  df = pd.read_csv(path)
  # print(df.head())
  # sys.exit()
  df["No"] = df["No"].astype(int)
  df["code"] = df["No"].apply(lambda x: "telm"+ str(x).zfill(3))
  df = df.set_index("code").T
  df = df.to_dict()
  return df

def near_rad(code):
  df = mk_rad_table()
  df["code2"] = df["code"].apply(lambda x: x[:4])
  df["code2"] = df["code2"].apply(lambda x: np.nan if x=="kans" or x=="kepv" else x)
  df = df.dropna()
  
  teleme_dict = load_teleme_position()
  lon,lat = teleme_dict[code]["lon"],teleme_dict[code]["lat"]
  
  df["distance"] = df[["lon","lat"]].apply(lambda x: calc_km(x[0],x[1],lon,lat),axis=1)
  df = df.sort_values("distance")
  point = df["code"].values[0]
  if point == 'unyo016':
    point = df["code"].values[1]
  return point


def snow_PV_rate(x,a=0.25,threshold=55):
  """[summary]
  Args:
      x ([type]): [snow depth]
      a (int, optional): [description]. Defaults to 55.
      threshold (float, optional): [description]. Defaults to 0.25.
  Returns:
      [type]: snow PV rate(0~1)
  """
  return 1/(1 + np.exp(a * (x - threshold )))

def pre_snow_depth_30min():
  path = "/work/ysorimachi/hokuriku/dat2/snow_jma/tmp/telm_snow_depth.csv"
  OUT_SNOW30 = "/work/ysorimachi/hokuriku/dat2/snow_jma/tmp/telm_snow_depth_30MIN.csv"
  #--------------------------
  def clensing(df):
    _df=[]
    _mm  = sorted(df["mm"].unique().tolist())
    
    for mm in _mm:
      df2 = df[df["mm"]==mm]
      n_dd = len(list(df2["dd"].unique())) 
      t = pd.DataFrame()
      t["time"] = pd.date_range(start=f"{mm}010000", freq="30T", periods=2*24*n_dd)
      t["_"] = 1
      t = t.set_index("time")
      df2 = pd.concat([t,df2],axis=1)
      
      df2 = df2.drop(["_","dd","mm"],axis=1)
      
      for c in df2.columns:
        df2[c] = df2[c].interpolate(method="linear")
      
      _df.append(df2)
    
    df = pd.concat(_df,axis=0)
    return df
  
  def preprocess(df):
    df = df.set_index("Unnamed: 0")
    df.index.name = "time"
    df = df.reset_index()
    df["time"] = pd.to_datetime(df["time"].astype(str))
    df["mm"] = df["time"].apply(lambda x: x.strftime("%Y%m"))
    df["dd"] = df["time"].apply(lambda x: x.strftime("%d"))
    df = df.set_index("time")
    return df
  #----------------------
  df = pd.read_csv(path)
  df = preprocess(df)
  df = clensing(df)
  df.to_csv(OUT_SNOW30)
  return 

def load_snowdepth(code="telm001"):
  OUT_SNOW30 = "/work/ysorimachi/hokuriku/dat2/snow_jma/tmp/telm_snow_depth_30MIN.csv"
  #-----------
  if not os.path.exists(OUT_SNOW30):
    pre_snow_depth_30min()
  #-----------
  df = pd.read_csv(OUT_SNOW30)
  df["time"]  = pd.to_datetime(df["time"])
  df = df.set_index("time")
  if type(code) ==list:
    return df[code]
  elif code is not None:
    return df[code]
  else:
    return df


def load_snowdepth_dd(dd="20210110"):
  OUT_SNOW30 = "/work/ysorimachi/hokuriku/dat2/snow_jma/tmp/telm_snow_depth_30MIN.csv"
  #-----------
  if not os.path.exists(OUT_SNOW30):
    pre_snow_depth_30min()
  #-----------
  df = pd.read_csv(OUT_SNOW30)
  df["time"]  = pd.to_datetime(df["time"])
  df["dd"] = df["time"].apply(lambda x: x.strftime("%Y%m%d"))
  df =df[df["dd"]==dd]
  return df


if __name__ == "__main__":
  
  # for col in ["tenminPrecip","temp","tenminSunshineTime","snowDepth"]:
  #   ave_data(col)
    
  # near_rad("telm001")
  # load_teleme_position()
  
  if 1:
    # snow data set
    # pre_snow_depth_30min()
    # df = load_snowdepth(code="telm001")
    load_snowdepth_dd()

# 'time', 'lat', 'lon', 'z', 'tenminPrecip', 'sixtyminPrecip',
#        'windDirection', 'windSpeed', 'temp', 'tenminMaxTemp', 'tenminMinTemp',
#        'tenminSunshineTime', 'snowDepth'
