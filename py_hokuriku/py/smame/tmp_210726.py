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
sys.path.append('/home/ysorimachi/work/hokuriku/py')
from utils import *


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




local = "/work/ysorimachi/hokuriku/dat2/snow_sfc/origin"
PWD="/home/ysorimachi/work/hokuriku/py/smame"

def download_sfc():
  _name = ["富山","福井"]
  _scode = [ acode2scode(acode=name2code(name)) for name in _name]
  _month = loop_month(st = "201904", ed="202104")
  
  for scode,name in zip(_scode,_name):
    for month in _month:
      if 0: #ftp get
        subprocess.run("sh ../sfc_get2.sh {} {} {}".format(month,scode,local), cwd=PWD,shell=True)
      
      path = f"{local}/sfc_10minh_{month}_{scode}.csv"
      df = pd.read_csv(path)
      df = conv_sfc(df,ave=30)
      print(df.head())
      sys.exit()
      print(datetime.now(), "[end]" , scode, name , month)
      # sys.exit()
  
  return



if __name__ == "__main__":
  
  # scode = acode2scode(acode=name2code("福井"))
  download_sfc()