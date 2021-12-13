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
sys.path.append("..")
from utils import *



INIT_DATA="/home/ysorimachi/work/hokuriku/dat/load/init_data"
OUT_DATA="/work/ysorimachi/hokuriku/dat2/load/set1"
def pre_data():
  def calc_hhmi(x,dd):
    hh = int(x//1)
    if hh !=24:
      mi = int((x-hh)*60)
      hh = str(hh).zfill(2)
      mi = str(mi).zfill(2)
      time = pd.to_datetime(f"{dd}{hh}{mi}")
      return time
    if hh ==24:
      time = pd.to_datetime(f"{dd}0000") + timedelta(days=1)
      return time
    

  _month = loop_month()
  _day = loop_day()
  for month,dd in zip(_month,_day):
    _dd = pd.date_range(start=f"{month}010000",periods=dd, freq="D")
    _dd =  [ d.strftime("%Y%m%d") for d in _dd]
    
    #------------------------
    _df=[]
    for dd in tqdm(_dd):
      path = f"{INIT_DATA}/{dd}_DAY_DATA.csv"
      if os.path.exists(path):
        subprocess.run(f"nkf -w --overwrite {path}", shell=True)
        # df = pd.read_csv(path,encoding="shift-jis")
        df = pd.read_csv(path)
        list1 = df.columns
        df = pd.read_csv(path,skiprows=1)
        df["時限"] = df["時限"].apply(lambda x: calc_hhmi(x,dd=dd))
        df = df.set_index("時限")
        df.index.name="time"
        # 'H900100'
        _col = [ f"H9{str(i).zfill(3)}00_{c}" for i in range(1,200+1) for c in ["IN","EX","PV"]]
        df.columns = _col
        _df.append(df)
      else:
        print(dd, "is Not Founded !")
    df= pd.concat(_df,axis=0)
    df.to_csv(f"{OUT_DATA}/{month}.csv")
    print(datetime.now(),"[END]", month)
    #------------------------
    # sys.exit()
  return


if __name__ == "__main__":
  if 0:
    pre_data()
    
  if 1:
    check_load()
