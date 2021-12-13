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
from utils import *
# os.makedirs(outd, exist_ok=True)

DHOME="/work/ysorimachi/hokuriku/dat2"

def loop_exel(cate):
  print(f"start {cate} ...")
  if cate == "teleme":
    ORIGIN="/home/ysorimachi/work/hokuriku/dat/teleme/re_get"
    os.makedirs(f"{DHOME}/teleme/pre", exist_ok=True)
    for month in loop_month():
      path = f"{ORIGIN}/PVSMP1min_{month}.xlsx"
      df = pd.read_excel(path,sheet_name= "Sheet1",engine='openpyxl',skiprows=1)
      df.to_csv(f"{DHOME}/teleme/pre/{month}.csv",index=False)
      print(datetime.now(),f"end {month} ...")
    return
  if cate == "smame":
    ORIGIN="/home/ysorimachi/work/hokuriku/dat/smame/re_get"
    os.makedirs(f"{DHOME}/smame/pre", exist_ok=True)
    for sub in ["全量","余剰"]:
      if sub == "全量":
        name = "all"
      else:
        name = "surplus"
      for month in loop_month():
        if int(month)<202004:
          yy = "2019"
        else:
          yy = "2020"
        path = f"{ORIGIN}/{yy}_{sub}/{month}_{sub}.xlsx"
        df = pd.read_excel(path,sheet_name= "Sheet1",engine='openpyxl',skiprows=1)
        df.to_csv(f"{DHOME}/smame/pre/{month}_{name}.csv",index=False)
        print(datetime.now(),f"end {month} ...")
    sys.exit()
    
def split_1day_smame(cate="all"):
  DHOME="/work/ysorimachi/hokuriku/dat2/smame/pre"
  OHOME="/work/ysorimachi/hokuriku/dat2/smame/dataset"
  # cate="all"
  
  for month in loop_month():
    df = pd.read_csv(f"{DHOME}/{month}_{cate}.csv")
    df = df.dropna(subset=["実績日"])
    df["実績日"] = df["実績日"].astype(int)
    # print(df.columns)
    # print(df['実績日'].unique())
    _dd = sorted(list(df['実績日'].unique()))
    
    for dd in _dd:
      tmp = df[df["実績日"]==dd]
      tmp.to_csv(f"{OHOME}/{cate}_{dd}.csv",index=False)
      # print(tmp.head())
      # print(tmp.dtypes)
      # sys.exit()
      print(datetime.now(), "[end]",cate, month,dd)
      # print(OHOME)
      # print("here is ",subprocess.run("pwd", shell=True))
  return 
  # sys.exit()
      
def clensing_smame(cate="all"):
  OHOME="/work/ysorimachi/hokuriku/dat2/smame/dataset"
  OHOME2="/work/ysorimachi/hokuriku/dat2/smame/dataset2"
  # cate = "all"
  _path = glob.glob(f"{OHOME}/{cate}_*.csv")
  for path in tqdm(_path):
    f_name = os.path.basename(path)
    df = pd.read_csv(path)
    
    #clensing --------------------------
    df = df.drop("実績日",axis=1)
    df = df.set_index("番号")
    df.index.name = "code"
    _col = [ c.split("～")[1].replace("：","") for c in df.columns]
    df.columns = _col
    
    # -----------------------------
    # 2021.11.30 update 
    for c in df.columns:
      df[c] = df[c].apply(lambda x: isFloat(x)) #float or 9999.
    # print(df.dtypes)
    # -----------------------------
    
    df.to_csv(f"{OHOME2}/{f_name}")
  return 
#-------------------------
def loop_dataset(cate):
  print(f"start {cate} ...")
  if cate == "teleme":
    DATASET = f"{DHOME}/teleme/pre"
    OUTD= f"{DHOME}/teleme/dataset"
    _month = loop_month()
    # _month=["202005"]
    _month=["202003"]
    
    for month in _month:
      path = f"{DATASET}/{month}.csv"
      # df = pd.read_excel(path,sheet_name= "Sheet1",engine='openpyxl',skiprows=1)
      df = pd.read_csv(path).drop(["年月"],axis=1)
      df = df.rename(columns = {"data_time": "time"})
      df["time"] = pd.to_datetime(df["time"])

      #3-----------------------------------------
      # date 2021.09.02
      # 202003 において、3/16日の00:00データが欠測しているので、補いつつ、30分平均の処理を行う
      if month == "202003":
        _t = pd.date_range(start=f"{month}160000",periods=5,freq="1T")
        tmp = pd.DataFrame()
        tmp["time"] = _t
        # for c in df.columns:
        #   tmp[c] = np.nan
        # tmp = tmp.set_index("time")
        # print(df.shape)
        # df = pd.concat([df,tmp],axis=0)
        df = df.merge(tmp, on="time",how="outer")
        # print(df.shape)
        # df = df.reset_index()
        df = df.sort_values("time").reset_index(drop=True)
      #3-----------------------------------------

      df = df.drop_duplicates(subset=["time"],keep="last") #2021.07.21
      # print(df.shape)
      # sys.exit()
      
      df = df.set_index("time")
      df = clensing_col(df, df.columns)
      
      _, rename_hash = mk_teleme_table()
      
      rename_col = [rename_hash[c].replace("\n","") for c in df.columns]
      df.columns = rename_col
      
      df = df.replace(9999,np.nan)
      # print(df.describe().T["max"].max())
      # sys.exit()
      
      df.to_csv(f"/home/ysorimachi/work/hokuriku/dat/rad/r2/rad/teleme_1min_tmp{month}.csv")
      df.to_csv(f"{OUTD}/1min_{month}.csv")
      for c in df.columns:
        df[c] = df[c].rolling(30).mean()
      df = df.iloc[::30]
      # print(df.head())
      df.to_csv(f"{OUTD}/30min_{month}.csv")
      df.to_csv(f"/home/ysorimachi/work/hokuriku/dat/rad/r2/rad/teleme_30min_tmp{month}.csv")
      # df.columns
      print(datetime.now(),f"end {month} ...")
      # sys.exit()
  
  if cate == "smame":
    for cate in ["all","surplus"]:
      split_1day_smame(cate=cate)
      clensing_smame(cate=cate)
    # sys.exit()
    return


#-----------------------------------
def read_origin(cate="teleme"):
  """swich function ..."""
  # loop_exel(cate) #originデータから一番最初に変換する
  loop_dataset(cate)
  return
  

if __name__ == "__main__":
  
  if 0:
    cate = "teleme"
    # cate = "smame"
    read_origin(cate=cate)
    # print(loop_month())
  
  if 1:
    # for cate in ["surplus"]:
    for cate in ["all"]:
      # split_1day_smame(cate=cate)
      clensing_smame(cate=cate)
  
  
