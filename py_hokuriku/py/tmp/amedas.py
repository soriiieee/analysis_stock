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

TBL_DIR="/home/ysorimachi/work/hokuriku/tbl/amedas"
use_col= ['観測所番号', '種類', '観測所名', 'カタカナ名','緯度(度)', '緯度(分)','経度(度)', '経度(分)']
# use_snow_col =['観測所番号', '種類', '観測所名', 'カタカナ名', '緯度(度)', '緯度(分)','経度(度)', '経度(分)']

def load_data(cate):
  path = glob.glob(f"{TBL_DIR}/{cate}_*.csv")[0]
  df = pd.read_csv(path)
  df =  df[use_col]
  return df

def load_rad_obs():
  path ="/home/ysorimachi/work/hokuriku/tbl/obspoint_rad48.dat"
  df = pd.read_csv(path,header=None,delim_whitespace=True)
  df = df[[0,1,2]]
  df.columns = ["scode","観測所番号","name"]
  return df

def calc_latlon(df):
  df["lat"] = df["緯度(度)"] + df["緯度(分)"]/60
  df["lon"] = df["経度(度)"] + df["経度(分)"]/60
  df = df.drop(['緯度(度)', '緯度(分)', '経度(度)', '経度(分)'],axis=1)
  
  return df


def get_List():
  """
  2021.09.09 統合コード
  """
  # rad-points
  rad = load_rad_obs()
  # data -points 
  df = load_data("ame")
  df2 = load_data("snow")
  df  =df.merge(df2,on=['カタカナ名','観測所名', '緯度(度)', '緯度(分)', '経度(度)', '経度(分)'], how="inner")
  df = df.merge(rad, left_on="観測所番号_x", right_on="観測所番号",how="left")
  # dropcol
  if 1:
    df = df.drop(['カタカナ名','観測所名','種類_y',"観測所番号"],axis=1)
    df = calc_latlon(df)
    use_col = ['観測所番号_x','観測所番号_y','種類_x','scode', 'name', 'lat', 'lon']
    rename_col = ['code','code_snow','cate','scode', 'name', 'lat', 'lon']
    df = df[use_col]
    df.columns = rename_col
  return df

def update():
  if 0: # download
    subprocess.run("sh ./get_ame.sh {}".format(TBL_DIR), shell=True)  
  return

if __name__ == "__main__":
  # update()
  get_amedas()