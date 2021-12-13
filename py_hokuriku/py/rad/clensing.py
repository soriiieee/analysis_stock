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
# outd ='/home/griduser/work/sori-py2/deep/out/0924_01'
# os.makedirs(outd, exist_ok=True)
# subprocess.run('rm -f *.png', cwd=outd, shell=True)



def rad_JOGAI(df,cate="obs"):
  """
  2021.11.03 clensingの処理の実施　-->
  """
  if not "dd" in df.columns:
    df["dd"] = df["time"].apply(lambda x: x.strftime("%Y%m%d"))
  
  if not "hh" in df.columns:
    df["hh"] = df["time"].apply(lambda x: x.hour )
   
  if cate =="obs":
    # unyo001 
    df.loc[df["dd"]=="20190418","unyo001"] = np.nan
    df.loc[df["dd"]=="20190801","unyo001"] = np.nan
    df.loc[df["dd"]=="20190904","unyo001"] = np.nan
    # unyo002
    df.loc[df["dd"]=="20190904","unyo002"] = np.nan
    df.loc[df["dd"]=="20210224","unyo002"] = np.nan
    df.loc[df["dd"]=="20210224","unyo002"] = np.nan
    df.loc[df["dd"]=="20210712","unyo002"] = np.nan
    # unyo003
    df.loc[df["dd"]=="20190603","unyo003"] = np.nan
    df.loc[df["dd"]=="20190604","unyo003"] = np.nan
    df.loc[df["dd"]=="20190605","unyo003"] = np.nan
    df.loc[df["dd"]=="20190606","unyo003"] = np.nan
    df.loc[df["dd"]=="20190620","unyo003"] = np.nan
    df.loc[df["dd"]=="20190904","unyo003"] = np.nan
    # unyo004
    df.loc[df["dd"]=="20190904","unyo004"] = np.nan
    # unyo005
    df.loc[df["dd"]=="20190904","unyo005"] = np.nan
    # unyo006(16時以降の時間帯が異常値なので、全部のデータで除外する)
    df.loc[df["hh"]>=16,"unyo006"] = np.nan
    # unyo007
    df.loc[df["dd"]=="20190930","unyo007"] = np.nan
    df.loc[df["dd"]=="20200228","unyo007"] = np.nan
    # unyo009
    df.loc[df["dd"]=="20191106","unyo009"] = np.nan
    df.loc[df["dd"]=="20191114","unyo009"] = np.nan
    df.loc[df["dd"]=="20200227","unyo009"] = np.nan
    # unyo009
    df.loc[df["dd"]=="20191128","unyo010"] = np.nan
    
    df.loc[df["dd"]=="20200219","unyo011"] = np.nan
    df.loc[df["dd"]=="20210615","unyo011"] = np.nan
    df.loc[df["dd"]=="20210625","unyo011"] = np.nan
    
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
    
  if cate =="8now0": #-----------------------------------
    _col = [ c for c in df.columns if "unyo" in c]
    for c in _col:
      df.loc[df["dd"]=="20200324",c] = np.nan
      df.loc[df["dd"]=="20210324",c] = np.nan
  
  df = df.drop(["dd","hh"],axis=1)
  df = df.round(3)
  df = df.replace(9999,np.nan)
  return df