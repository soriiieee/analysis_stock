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
  _t = _t[:-1]
  return _t



def rad_code2name(code="unyo001"):
  if code == "mean" or code == "ave":
    return "エリア平均"
  path = f"/home/ysorimachi/work/hokuriku/dat/rad/re_get/list_rad_point2.csv"
  df = pd.read_csv(path).set_index("code")
  name_dict = df["name"].to_dict()
  
  if code=="kans001":
    return "Toyama-KAN"
  elif code=="kans002":
    return "Fukui-KAN"
  elif code =="ave":
    return "-"
  else:  
    return name_dict[code]


def mk_rad_table():
  path = f"/home/ysorimachi/work/hokuriku/dat/rad/re_get/list_rad_point2.csv"
  df = pd.read_csv(path)
  # .set_index("code")
  return df

def mk_teleme_table():
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
  return df, rename_hash
    # sys.exit()



if __name__ == "__main__":
  # df, rename_hash = mk_teleme_table()
  # rad_code2name()
  dat2csv_8now()