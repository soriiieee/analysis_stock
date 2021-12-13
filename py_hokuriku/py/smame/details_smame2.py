# -*- coding: utf-8 -*-
# when   : 2021.09.12
# when   : 2021.11.15 pv-mesh　の作成を行う予定
# who : [sori-machi]
# what : [ smame ようのもづーる]
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
#---
import matplotlib.colors as mcolors
_color = [ mcolors.rgb2hex(plt.cm.get_cmap('tab10')(i)) for i in range(10)]
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
#---------------------------------------------------
# sori -module
sys.path.append('/home/ysorimachi/tool')
from getErrorValues import me,rmse,mae,r2 #(x,y)
from convSokuhouData import conv_sfc #(df, ave=minutes,hour)
from distance import calc_km #"(lon_a,lat_a,lon_b,lat_b)
#amedas relate 2020.02,04 making...
# from tool_AMeDaS import code2name, name2code
# from tool_110570 import get_110570,open_110570
# from tool_100571 import get_100571,open_100571
#(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
import subprocess
from mapbox import map_lonlat#(df,html_path,text="name",size=4,size_col=None,zoom=4) 


sys.path.append('/home/ysorimachi/work/hokuriku/py')
from utils import *

SMAME="/work/ysorimachi/hokuriku/dat2/smame/dataset2"
SMAME_TBL="/home/ysorimachi/work/hokuriku/tbl/smame"
RAD_LIST ="/home/ysorimachi/work/hokuriku/dat/rad/re_get/list_rad_point2.csv"
RAD_LIST2 ="/home/ysorimachi/work/hokuriku/dat/rad/re_get/list_rad_point_near.csv"

def load_smame_list(cate=None):
  path = "../../tbl/smame/list_smame.csv"
  df = pd.read_csv(path)
  df["cate"] = df["code"].apply(lambda x: x[0])
  
  if cate == "surplus" or cate=="Y":
    df = df[df["cate"]=="Y"]
  elif cate == "all" or cate=="Z":
    df = df[df["cate"] == "Z"]
  else:
    pass
  return df


#------------2021.10.08 --------------
def save_numpy(save_path,obj):
  save_path2 = save_path.split(".")[0]
  np.save(save_path2, obj.astype('float32'))
  return 

def load_numpy(path):
  obj = np.load(path)
  return obj

OUT_HOME = "/home/ysorimachi/work/hokuriku/dat/snow/csv"
def smame_index(cate,center, area):
  center_lon,center_lat = center
  df = load_smame_list(cate) # get list smame (per　yojousumame)
  df = df[(df["lon"]>center_lon-area)&(df["lon"]<center_lon+area)&
          (df["lat"]>center_lat-area)&(df["lat"]<center_lat+area)]
  
  if 0:
    html_path = "/home/ysorimachi/work/hokuriku/dat/snow/csv/mapbox/tmp.html"
    map_lonlat(df,html_path,text="code",size=10,size_col="max[kW]",zoom=6)
    
  df.to_csv(f"{OUT_HOME}/select/smame.csv", index = False)
  
  _index =df["code"].values.tolist()
  return _index

def load_smame_dd(cate, day):
  path = f"{SMAME}/{cate}_{day}.csv"
  df = pd.read_csv(path)
  return  df

def smame_max(cate, _code):
  df = load_smame_list(cate)
  df = df.loc[df["code"].isin(_code), :]
  return df["max[kW]"].sum()

  

def sub_smame_dataset(cate="surplus"):
  center = [136.608551,36.37216]
  _code = smame_index(cate,center,area=0.05)
  
  _month = loop_month(st = "201904", ed="202104")
  _dd = loop_day(st = "201904", ed="202104")
  def clensing(df):
    for c in df.columns:
      df[c] = df[c].apply(lambda x: isFloat(x))
    return df
  for mm,dd_max in zip(_month,_dd):
    _day = pd.date_range(start = f"{mm}010000", periods=dd_max, freq="D")
    _day = [ d.strftime("%Y%m%d") for d in _day]
    # print(_day[-1])
    _df=[]
    for day in _day:
      df = load_smame_dd(cate = cate, day=day)
      df = df.loc[df["code"].isin(_code), :]
      use_code = df["code"].values.tolist()
      pv_max = smame_max(cate, use_code)
      df = df.set_index("code").T
      df.index.name = "HHMM"
      
      # preprocessing ---
      df = clensing(df)
      df["sum"] = df.sum(axis=1)
      df["count"] =df.count(axis=1)
      df = df.reset_index()
      df["max"] = pv_max
      df = df[["HHMM","sum","count","max"]]
      df["time"] = df["HHMM"].apply(lambda x: f"{day}{x}")
      _df.append(df)
      print(datetime.now(),"[end]", day)
    df = pd.concat(_df,axis=0)
    df.to_csv(f"{OUT_HOME}/tmp_smame_month/{cate}_{mm}.csv", index=False)
    
  return
    

def pv_mesh(cate, reset=False):
  df = load_smame_list(cate)
  
  if reset:
    mesh = np.zeros((7200,4800))
    _lon = np.linspace(120,150,4800)
    _lat = np.linspace(20,50,7200)
    for i,r in df.iterrows():
      ix = np.argmin(np.abs(_lon - r.lon)) 
      iy = np.argmin(np.abs(_lat - r.lat))
      mesh[iy,ix] += r["max[kW]"]
    
    save_numpy(f"/home/ysorimachi/work/hokuriku/tbl/smame/mesh/mesh_{cate}.npy",mesh)
    # print(np.nanmax(mesh), np.nanmin(mesh))
    print(datetime.now(), "[end]",cate,df.shape)
  else:
    mesh = load_numpy(f"/home/ysorimachi/work/hokuriku/tbl/smame/mesh/mesh_{cate}.npy")
  return mesh

def load_rad_point():
  rad = pd.read_csv(RAD_LIST)
  rad["cate"] = rad["code"].apply(lambda x: x[:4])
  rad = rad[rad["cate"] != "kepv"]
  rad = rad[rad["code"] != "unyo016"]
  rad = rad.reset_index(drop=True)
  return rad

def near_rad_smame(cate="all"):
  out_path = "../../tbl/smame/list_smame_near_rad.csv"
  df = load_smame_list()
  # print(df.shape)
  # sys.exit()
  
  def calc_near(pos):
    rad = load_rad_point()
    rad["distance"] = rad[["lon","lat"]].apply(lambda x: calc_km(x[0],x[1],pos[0],pos[1]), axis=1)
    rad = rad.sort_values("distance")
    code = rad["code"].values[0]
    dis = rad["distance"].values[0]
    return code, dis
  
  #------------------------
  # df = df.iloc[::]
  _near =[]
  _dis = []
  for (i,r) in tqdm(list(df.iterrows())):
    pos = [r.lon,r.lat]
    code,dis = calc_near(pos)
    _near.append(code)
    _dis.append(dis)
  
  df["near_rad"] = _near
  df["near_km"] = _dis
  df.to_csv(out_path,index=False)
  return
  


if __name__ == "__main__":
  
  #--------------
  if 0: #2021.11.30 
    # smame list(with nearest RAD point ..)
    near_rad_smame("all")
  
  #--------------
  for cate in ["Z","Y","ALL"]:
    pv_mesh(cate)
  