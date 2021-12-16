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
outd ='/home/griduser/work/sori-py2/deep/out/0924_01'
# os.makedirs(outd, exist_ok=True)
# subprocess.run('rm -f *.png', cwd=outd, shell=True)


DIR= "/work2/ysorimachi/synfos/data/SYNFOS-solar_ver2/maeda_210709"
# mix1_col = ["time","MIX1","SYN","ECs0","ECCs0","fecm","fecc","rCR0","rS0"]
# mix2_col = ["time","MIX2","SYN","ECcr0","ECCcr0","fecm","fecc","rCR0","rS0"]

def load_rad(code,dd,mtd):
  if mtd==1:
    names = ["time","MIX1","SYN","ECs0","ECC","fecm","fecc","rCR0","rS0"]
  else:
    names = ["time","MIX2","SYN","ECcr0","ECC","fecm","fecc","rCR0","rS0"]
    
  path = f"{DIR}/{mtd}/{dd}/{code}.dat"
  df = pd.read_csv(path, header=None,delim_whitespace=True,names =names)
  df = df.drop(["ECC","fecm","fecc"],axis=1)
  if mtd==2:
    df = df.drop(["SYN","rCR0","rS0"],axis=1)
  return df

def load_code():
  tbl_path ="/home/ysorimachi/work/ecmwf/tbl/list_pnt_210709.tbl"
  df = pd.read_csv(tbl_path, delim_whitespace=True, header=None)
  _code = df[0].values.tolist()
  return _code

def data():
  _dd = sorted(os.listdir(f"{DIR}/1"))
  _code = load_code()
  
  log_file = "./data.log"
  # subprocess.run("rm -rf {}".format(log_file), shell=True)
  
  for code in _code:
    for dd in _dd:
      #init director ----
      outd=f"{DIR}/store/{dd}"
      os.makedirs(outd, exist_ok=True)
      
      #----make concat ----
      try:
        r1 = load_rad(code,dd,1)
        r2 = load_rad(code,dd,2)
        df = r1.merge(r2,on="time",how="inner")
        use_col = ["time","SYN","ECs0","ECcr0","MIX1","MIX2","rS0","rCR0"]
        df = df[use_col]
        # print(df.replace(9999,np.nan).describe())
        df.to_csv(f"{outd}/{code}.csv", index=False)
      # print(outd)
        with open(log_file, "+a") as f:
          now=datetime.now()
          text = f"{now} [--ok-] {code} {dd}\n"
          f.write(text)
      except:
        with open(log_file, "+a") as f:
          now=datetime.now()
          text = f"{now} [error] {code} {dd}\n"
          f.write(text)
          
      #----make concat ----
  # sys.exit()
  return

if __name__ == "__main__":
  # for mtd in [1,2]:
  data()
  # load_code()

