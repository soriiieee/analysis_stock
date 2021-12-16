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

# MODEL2="2model_ver6_20190215_kuno"
MODEL2="2model_ver7_20200226_kuno"
# MODEL3="3model_ver6_20190215_kuno"
MODEL3="3model_ver7_20200226_kuno"
TOGO="togo_ver1_20181015_obara"
path = "/home/ysorimachi/work/ecmwf/tbl/list_enrs_210728.tbl"

def load_tbl():
  df = pd.read_csv(path, delim_whitespace=True,header=None)
  df["code"] = df[19].astype(str)
  _code =df["code"].values.tolist()
  # print(df.head())
  return _code

def get_dir(cate):
  if cate=="2":
    return MODEL2
  if cate=="3":
    return MODEL3
  if cate=="t":
    return TOGO
  

def get_mos_code(cate="2"):
  sub_dir= get_dir(cate)
  DIR=f"/home/ysorimachi/work/synfos/mos/{sub_dir}"
  _f = os.listdir(DIR)
  _code = np.unique(sorted([ f[:5] for f in _f]))
  print("mos category -> ", cate)
  print(_code)
  print("-"*50)
  return _code

def main():
  _code = load_tbl()
  
  _c2 = get_mos_code(cate="2")
  _c3 = get_mos_code(cate="3")
  _ct = get_mos_code(cate="t")
  
  _code = np.unique(sorted(_code))
  fhash = {}
  for code in _code:
    f2 = code in _c2
    f3 = code in _c3
    ft = code in _ct
    fhash[code] = [f2,f3,ft]
  
  df = pd.DataFrame(fhash).T
  df.columns = ["mos2","mos3","togo"]
  df.index.name = "kansho"
  df.to_csv("./check_mos.csv")
    # print(f2)
    # sys.exit()
    

if __name__ == "__main__":
  main()
  # load_tbl()