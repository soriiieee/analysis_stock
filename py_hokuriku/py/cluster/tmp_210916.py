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
from plotPoint import plot_map#(df,params,png_path,isColor=False):
"""
edit-day : 2021.03.15
edit-day : 2121.06.20

〇色付きカラーマップ
input: df(pandas->DataFrame)： "lon","lat","z"(色付き)
〇通常プロット
input: df(pandas->DataFrame)： "lon","lat","z"があると色付きなので事前にdropしておく

:params["setmap"](list) : [lon_min,lon_max, lat_min, lat_max]
:params["cmap"]("String") : "jet"
"""

def sample():
  path = "/home/ysorimachi/work/8now_cast/dat/t_cluster/cluster/cluster_sum2_1029_sort.csv"
  df = pd.read_csv(path)
  df = df[["lon","lat","wald_20"]]
  df.columns = ["lon","lat","z"]
  # print(df.describe())
  # sys.exit()
  
  png_path = "/home/ysorimachi/work/hokuriku/out/cluster/sample.png"
  params = {
    "setmap" : [139,141.5,34.5,37],
    "cmap" : "jet"
  }
  plot_map(df,params,png_path,isColor=False)
  
if __name__ == "__main__":
  sample()

