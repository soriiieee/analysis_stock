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
# from getErrorValues import me,rmse,mae,r2 #(x,y)
# from convSokuhouData import conv_sfc #(df, ave=minutes,hour)
# #amedas relate 2020.02,04 making...
# from tool_AMeDaS import code2name, name2code
# from tool_110570 import get_110570,open_110570
# from tool_100571 import get_100571,open_100571
# #(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
import subprocess
# outd ='/home/griduser/work/sori-py2/deep/out/0924_01'
# os.makedirs(outd, exist_ok=True)
# subprocess.run('rm -f *.png', cwd=outd, shell=True)


def load_amazon_twitter():
  url = "https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv"
  df = pd.read_csv(url, header=0, index_col=0)
  df = df.reset_index()
  df["timestamp"] = pd.to_datetime(df["timestamp"])
  df.columns = ["time","value"]
  df = df.set_index("time")
  return df


def load_power_price(point="hkdn001"):
  url = "/home/ysorimachi/work/pp_fct/dat/tmp/all_dataset.csv"
  df = pd.read_csv(url)
  df["time"] = pd.to_datetime(df["time"])
  df = df[["time",point]]
  # df["timestamp"] = pd.to_datetime(df["timestamp"])
  df.columns = ["time","value"]
  df = df.set_index("time")
  return df

# df = load_power_price()
