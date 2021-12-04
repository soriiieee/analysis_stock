# -*- coding: utf-8 -*-
# when   : 2020.0x.xx
# who : [sori-machi]
# what : [ ]
#---------------------------------------------------------------------------
# basic-module
# import matplotlib.pyplot as plt
import sys,os,re,glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.simplefilter('ignore')
# from tqdm import tqdm
# import seaborn as sns
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
# #---------------------------------------------------
# import subprocess
# outd ='/home/griduser/work/sori-py2/deep/out/0924_01'
# os.makedirs(outd, exist_ok=True)
# subprocess.run('rm -f *.png', cwd=outd, shell=True)
import pygrib


def get_ixiy(nx,ny,lon0,lon1,lat0,lat1,rlon,rlat):
  """
  2021.02.28
  ecmwf データの緯度経度からpointのｉｘ，ｉｙを抽出するプログラム
  """
  rdx = (lon1 -lon0) / nx
  rdy = (lat0 -lat1) / ny

  ilon = int(np.floor((rlon-lon0)/rdx))
  ilat = int(np.floor((lat0-rlat)/rdy))
  return ilon,ilat

def get_ixiy2(info,rlon,rlat):
  """
  2021.02.28
  ecmwf データの緯度経度からpointのｉｘ，ｉｙを抽出するプログラム
  """
  
  lons = np.array(info["distinctLongitudes"])
  lats = np.array(info["distinctLatitudes"])
  
  ilon = np.argmin(np.abs(lons - rlon))
  ilat = np.argmin(np.abs(lats - rlat))
  
  return ilat,ilon


def get_info(path):
  """
  2021.02.28
  ecmwf data のsetting情報を表示する関数
  """
  gribs = pygrib.open(path)
  info= gribs.select()[0]
  
  lon0 = info['longitudeOfFirstGridPointInDegrees']
  lon1 = info['longitudeOfLastGridPointInDegrees']
  lat0 = info['latitudeOfFirstGridPointInDegrees']
  lat1 = info['latitudeOfLastGridPointInDegrees']
  
  nx = info['Ni']
  ny = info['Nj']
  return nx,ny,lon0,lon1,lat0,lat1

def get_info2(path):
  """
  2021.02.28
  ecmwf data のsetting情報を表示する関数
  """
  gribs = pygrib.open(path)
  info= gribs.select()[0]
  return info

def ensemble_ecmwf(path,cele='Surface solar radiation downwards'):
  """
  アンサンブルデータの平均と偏差を取得する関数
  cele の一覧
  https://apps.ecmwf.int/codes/grib/param-db
  """
  # nx,ny,lon0,lon1,lat0,lat1 = get_info(path)
  gribs = pygrib.open(path)
  _fct=[]
  # value = np.array()
  _value = []
  for i,grb in enumerate(list(gribs)):
    if grb["parameterName"]==cele:
      _value.append(grb.values)
    else:
      pass
  
  value = np.array(_value)
  mesh_mean = np.mean(value,axis=0)
  mesh_std = np.std(value,axis=0)
  # print(mean.shape,std.shape)
  return mesh_mean,mesh_std


def all_ecmwf(path,cele='Surface solar radiation downwards'):
  """
  アンサンブルデータの平均と偏差を取得する関数
  cele の一覧
  https://apps.ecmwf.int/codes/grib/param-db
  """
  # nx,ny,lon0,lon1,lat0,lat1 = get_info(path)
  gribs = pygrib.open(path)
  _fct=[]
  # value = np.array()
  _value,_n_ens = [],[]
  for i,grb in enumerate(list(gribs)):
    if grb["parameterName"]==cele:
      _value.append(grb.values)
      _n_ens.append(grb["perturbationNumber"])
    else:
      pass
  
  value = np.array(_value)
  n_ens = np.array(_n_ens)
  return value,n_ens


  grb["perturbationNumber"]
  return 

def get_ecmwf_details(path):
  gribs = pygrib.open(path)
  info= gribs.select()[0]
  # for key in info.keys():
    # print(key,info[key])
  
  _col=[ "latLonValues","latitudes","longitudes","distinctLatitudes","distinctLongitudes","values"]
  for col in _col:
    print(col , info[col].shape)
    
  return 


if __name__=="__main__":
  path = "/home/ysorimachi/data/ecmwf/dat/D1E/2018033012/D1E03301200040115001"
  # get_info(path)
  info = get_info2(path)
  ilat,ilon  =get_ixiy2(info,131,35)
  print(ilat,ilon)
  sys.exit()
  get_ecmwf_details(path)