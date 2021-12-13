# -*- coding: utf-8 -*-
# when   : 2020.0x.xx
# who : [sori-machi]
# what : [ ]
#---------------------------------------------------------------------------
# basic-module
import sys,os,re,glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.simplefilter('ignore')
from tqdm import tqdm
import seaborn as sns


#(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
import subprocess
import pygrib
import cartopy.crs as ccrs
import cartopy.feature as cfea
# import cartopy.feature as cfea
import cartopy.io.shapereader as shapereader
from cartopy.feature import ShapelyFeature

from netCDF4 import Dataset
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors

from PIL import Image

def conv2utc(ctime,ctype="J"):
  if ctype=="J":
    get_time = (pd.to_datetime(ctime) + timedelta(hours=9)).strftime("%Y%m%d%H%M")
  if ctype=="U":
    get_time = (pd.to_datetime(ctime) - timedelta(hours=9)).strftime("%Y%m%d%H%M")
  return get_time

def get_lonlat(name="富山"):
  df=pd.read_csv("../../tbl/ame_master3.csv")
  try:
    lon,lat = list(df.loc[df["観測所名"]==name,["経度","緯度"]].values[0])
  except:
    lon,lat = 9999,9999
  return lon,lat

def  save_depth(img2,dx,dy,ini_j,_list):
  _val,_lon,_lat,_name = [],[],[],[]
  for name in _list:
    lon,lat = get_lonlat(name=name)
    if (135.5 < lon < 138) and (35.5 < lat < 38):
      jx,jy = int((lon-120)/dx) ,int((lat-20)/dy)
      # jx = np.argmin(np.abs(np.array(nc.variables["longitude"]) -lon))
      # jy = np.argmin(np.abs(np.array(nc.variables["latitude"]) - lat))
      val = img2[jy,jx]

      _val.append(val)
      _lon.append(lon)
      _lat.append(lat)
      _name.append(name)

  df = pd.DataFrame()
  df["name"] = _name
  df["lon"] = _lon
  df["lat"] = _lat
  df["val"] = _val
  df.to_csv(f"/home/ysorimachi/work/hokuriku/out/snow2/csv/121211_{ini_j}.csv", index=False,encoding="shift-jis")
  return



def main(ini_u):
  path = f"/home/ysorimachi/work/hokuriku/dat/snow/121211/121211-0000SP-0000-{ini_u}00.png"
  ini_u = ini_u
  ini_j = conv2utc(ini_u,ctype="J")

  projection = ccrs.PlateCarree()
  color_polygon = "k"
  lakes_ = cfea.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor=color_polygon, facecolor="none", linewidth=0.5)
  states_ = cfea.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '10m', edgecolor=color_polygon, facecolor='none', linewidth=0.2)

  img = np.array(Image.open(path))
  dx ,dy= 30./ img.shape[1], 30./ img.shape[0]

  lon0,lon1,lat0,lat1 = 135.5,138.0, 35.5, 38.0
  ix0,ix1 = int((lon0-120)/dx) ,int((lon1-120)/dx)
  iy0,iy1 = int((lat0-20)/dy) ,int((lat1-20)/dy)


  df = pd.DataFrame(img)
  img2 = np.flipud(df.replace(255,np.nan))
  # img2 = np.flipud(df.replace(0,0.001))
  img2 = img2[iy0:iy1,ix0:ix1]

  _list = ["氷見","富山","朝日","魚津","珠洲","輪島","七尾","金沢","白山河内","加賀菅谷","大野","福井","敦賀","小浜"]
  # save_depth(img2,dx,dy,ini_j,_list)
  # print(img2[-6:-1,-6:-1])
  # print(img2[:6,:6])
  # print(np.max(img2))
  # sys.exit()

  f = plt.figure(figsize=(10,10))
  ax = f.add_subplot(1,1,1,projection=projection)
  grd = ax.gridlines(crs=projection)
  
  # norm = mcolors.DivergingNorm( vcenter=15, vmin=0., vmax=300.0 )
  norm = mcolors.SymLogNorm( linthresh=1.0, linscale=1.0, vmin=0.1, vmax=500.0 )
  map0 = ax.imshow(img2, extent=(lon0,lon1,lat0,lat1), transform=projection, interpolation='nearest', cmap="cool", norm=norm)
  
  ax.set_extent((lon0,lon1,lat0,lat1),projection)
  ax.coastlines(resolution="10m")
  ax.add_feature(lakes_)
  ax.add_feature(states_)
  # ax.add_feature(cfea.OCEAN, zorder=100, edgecolor="k")
  # plt.imshow( arr, cmap='bwr', norm=norm )
  pp = plt.colorbar(map0,pad=0.03, fraction=0.03,shrink=0.7,extend="neither")
  # cbar = fig.colorbar(im, ticks=[0,0.5, 1])
  # pp.ax.set_ylim(iy0,iy1)
  _list = ["氷見","富山","朝日","魚津","珠洲","輪島","七尾","金沢","白山河内","加賀菅谷","大野","福井","敦賀","小浜"]
  for name in _list:
    lon,lat = get_lonlat(name=name)
    if (135.5 < lon< 138) and (35.5<lat<38):
      ax.plot(lon,lat)
      ax.text(lon,lat,name) 

  pp.ax.set_ylabel("snowdepth[cm]")  # vertically oriented colorbar    # ax.set_ylim(0,50)
  ax.set_title(f"121211(JST={ini_j})", loc="left",pad=None)
  plt.savefig(f"/home/ysorimachi/work/hokuriku/out/snow/121212_{ini_j}.png", bbox_inches="tight")
  plt.close()
  return

if __name__ =="__main__":
  ini_u = sys.argv[1]
  main(ini_u)