# -*- coding: utf-8 -*-
# when   : 2021.03.10 init
# when   : 2021.08.31 update
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


sys.path.append('/home/ysorimachi/tool')
from tool_cmap import cmap_snow
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

#(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
import subprocess
import pygrib
import cartopy.crs as ccrs
import cartopy.feature as cfea
# import cartopy.feature as cfea
import cartopy.io.shapereader as shapereader
from cartopy.feature import ShapelyFeature

from cartopy_land_mask import mask #_latlon=[120,150,20,50], mask_path

from netCDF4 import Dataset
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import copy

sys.path.append("/home/ysorimachi/work/hokuriku/py")
# from smame.eda_map import smame_position #(cate ="all", dd8="20200330")
# from utils import load_teleme_position
from tool_time import dtinc
import subprocess

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


def  save_depth(nc,ini_j,_list):
  _val,_lon,_lat,_name = [],[],[],[]
  for name in _list:
    lon,lat = get_lonlat(name=name)
    if (135.5 < lon< 138) and (35.5<lat<38):
      jx = np.argmin(np.abs(np.array(nc.variables["longitude"]) -lon))
      jy = np.argmin(np.abs(np.array(nc.variables["latitude"]) - lat))
      val = nc.variables["var0_1_232_surface"][0,jy,jx]*100

      _val.append(val)
      _lon.append(lon)
      _lat.append(lat)
      _name.append(name)

  df = pd.DataFrame()
  df["name"] = _name
  df["lon"] = _lon
  df["lat"] = _lat
  df["val"] = _val
  df.to_csv(f"/home/ysorimachi/work/hokuriku/out/snow2/csv/111095_{ini_j}.csv", index=False, encoding="shift-jis")
  return


def main(ini_u, cate=None,size=None,vmin=0.1,vmax=500):
  """
  init: 2021.02.20
  update: 2021.03.01 
  [main program]
  convert grib type data to netcdf datatyepes and use "Dataset" python libraly which convert to pandas df for plot map and timeseries datas asnd so on..  after get grib files
  
  """
  path = f"/home/ysorimachi/work/hokuriku/dat/snow/111095/{ini_u}.nc"
  ini_u = os.path.basename(path).split(".")[0]
  ini_j = conv2utc(ini_u,ctype="J")

  nc = Dataset(path,"r")

  # lon0,lon1,lat0,lat1 = 135.5,137.8,35.5,37.8
  ix0 = np.argmin(np.abs(np.array(nc.variables["longitude"]) - 135.5))
  ix1 = np.argmin(np.abs(np.array(nc.variables["longitude"]) - 138.0))

  iy0 = np.argmin(np.abs(np.array(nc.variables["latitude"]) - 35.5))
  iy1 = np.argmin(np.abs(np.array(nc.variables["latitude"]) - 38.0))

  xx = nc.variables["longitude"][ix0:ix1]
  yy = nc.variables["latitude"][iy0:iy1]
  data = nc.variables["var0_1_232_surface"][0,iy0:iy1,ix0:ix1]*100
  
  
  # https://cloud6.net/so/python/3050836
  data  = np.array(data)
  data = np.where(data == 0, np.nan, data)
  data = np.where(data == 9.999000e+20,np.nan,data)
  # print(data.shape)

  # mask_path = "/home/ysorimachi/work/hokuriku/out/snow/mask/hokuriku_area.png"
  # if not os.path.exists(mask_path):
  #   print("make mask ...")
  #   lon0,lon1 = np.min(xx),np.max(xx)
  #   lat0,lat1 = np.min(yy),np.max(yy)
  #   _latlon = [lon0,lon1,lat0,lat1]
  #   mask(_latlon,mask_path)

  def plot_map():
    """
    [suboroutine program]
    
    """
    lon0,lon1 = np.min(xx),np.max(xx)
    lat0,lat1 = np.min(yy),np.max(yy)
    projection=ccrs.PlateCarree()
    color_polygon = "k"
    lakes_ = cfea.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor=color_polygon, facecolor="none", linewidth=0.5) #facecolor="none"
    
    states_ = cfea.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '10m', edgecolor=color_polygon, facecolor="none", linewidth=0.2)
    lands_ = cfea.NaturalEarthFeature('physical', 'land', '10m')
    # ocean_ = cfea.NaturalEarthFeature('physical', 'ocean', '10m')
    f = plt.figure(figsize=(8,8))
    rcParams["font.size"] = 18
    ax = f.add_subplot(1,1,1,projection=projection)
    # ax.add_feature(lands_, color='gray')
    ax.set_extent((lon0,lon1,lat0,lat1),projection)
    ax.coastlines(resolution="10m")
    ax.add_feature(states_)
    # grd = ax.gridlines(crs=projection,linewidth=1, linestyle=':', color='gray', alpha=0.4)
    # grd.xlocator = mticker.FixedLocator(list(np.arange(lon0,lon1+0.1,1)))
    # grd.ylocator = mticker.FixedLocator(list(np.arange(lat0,lat1+0.1,1)))
    """
    外部サイト
    # https://cloud6.net/so/python/3050836
    # https://www.naturalearthdata.com/features/
    # cfea.COLORS['land']
    # ax.add_feature(cfeature.OCEAN, color='aqua') # 海を水色で塗り潰す
    # ax.add_feature(cfeature.LAKES, color='aqua') # 湖を水色で塗り潰す
    """
    
    #plot contour ----------------------------
    ax = plot_contour(ax,data, projection,vmin=vmin, vmax=vmax,extent=(lon0,lon1,lat0,lat1))
    
    #plot-----テレメとか-------------------------------
    # if not cate is None:
    ax = plot_sub_sites(ax,size=size)
    # sys.exit()
    # #plot-----官署-------------------------------
    # _list = ["福井","金沢","砺波","富山","魚津"]
    # ax = plot_site_text(ax,_list)
    
    #--------------------------------------------------
    ax.set_title(f"積雪深[cm](JST={ini_j})", loc="left",pad=None)
    png_path = f"/home/ysorimachi/work/hokuriku/out/snow/tmp/111095_{ini_j}_{cate}.png"
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    
  plot_map()
  return

def get_latlon(df,lon_col,lat_col):
  _lon = df[lon_col].values.tolist()
  _lat = df[lat_col].values.tolist()
  return _lon,_lat


def plot_sub_sites(ax,size=2):
  # if cate =="none":
  #   return ax
  # if cate =="s_smame":
  #   df = smame_position("surplus","20200331")
  #   _lon,_lat = get_latlon(df,"lon","lat")
  # if cate =="a_smame":
  #   df = smame_position("all","20200331")
  #   _lon,_lat = get_latlon(df,"lon","lat")
  # if cate =="teleme":
  #   df = load_teleme_position()
  #   _lon,_lat = get_latlon(df,"lon","lat")
  
  """
  2021.9.1
  """
  df= mk_rad_table()
  df["cate"] = df["code"].apply(lambda x: x[:4])
  df= df[(df["cate"]=="kans")|(df["cate"]=="unyo")]
  _lon = df["lon"].values.tolist()
  _lat = df["lat"].values.tolist()
  size=80
  ax.scatter(_lon,_lat,marker="^",s=size, color="green", alpha=1)
  return ax

def plot_contour(ax,data,projection,vmin,vmax,extent=(120,150,20,50)):
  # Image.paste(im, box=None, mask=None)
  # norm = mcolors.SymLogNorm(linthresh=1.0, linscale=1.0, vmin=vmin, vmax=vmax)
  cmap,norm = cmap_snow()
  # map0 = ax.imshow(data,extent=extent, transform=projection, cmap="cool",norm=norm) #vmin,vmax
  map0 = ax.imshow(data,extent=extent, transform=projection, cmap=cmap,norm=norm, vmin=vmin,vmax=vmax) #vmin,vmax
  
  pp = plt.colorbar(map0,pad=0.03, fraction=0.03, shrink=0.9,extend="neither")
  pp.ax.set_ylabel("snowdepth[cm]")  # vertically oriented colorbar    # ax.set_ylim(0,50)
  return ax

def plot_site_text(ax,_list):
  for name in _list:
    lon,lat = get_lonlat(name=name)
    if (135.5 < lon< 138) and (35.5<lat<38):
      ax.scatter(lon,lat,marker="v",s=30, color="r")
      ax.text(lon,lat,name,fontsize=15,color="k")
  return ax

#---------------
# 2021.08.31 に作成するprogram立ち
def get_bin_file(ini_u):
  local = "/home/ysorimachi/work/hokuriku/dat/snow/111095"
  subprocess.run(f"rm {local}/*.bin",shell=True)
  subprocess.run(f"rm {local}/*.nc",shell=True)
  subprocess.run("sh get_kaiseki_snow.sh {} {}".format(ini_u,local) ,shell=True)

def mk_rad_table():
  path = f"/home/ysorimachi/work/hokuriku/dat/rad/re_get/list_rad_point2.csv"
  df = pd.read_csv(path)
  # .set_index("code")
  return df
#---------------



if __name__ =="__main__":
  # ini_u = sys.argv[1]
  # ini_j = "202101251200"
  _ini_j = [ "202101111200","202101201200","202101251200"]
  
  # _size = [1,1,1,15]
  # _cate = ["none","s_smame","a_smame","teleme"]
  
  _size = [1]
  _cate = ["none"]

  for ini_j in _ini_j:
    ini_u = dtinc(ini_j, 4,-9)
    if 1:
      get_bin_file(ini_u)
    
    if 1:
      for size, cate in zip(_size, _cate):
        main(ini_u, cate=cate,size=size)
        print("end", cate,"....")
  
  # plot_sub_sites("teleme")
  
  # concat_2img("/home/ysorimachi/work/hokuriku/out/snow/111095_202002010000.png","/home/ysorimachi/work/hokuriku/out/snow/mask/hokuriku_area.png","/home/ysorimachi/work/hokuriku/out/snow/concat/111095_202002010000.png")
  