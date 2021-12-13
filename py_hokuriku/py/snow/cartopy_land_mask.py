# -*- coding: utf-8 -*-
import sys,os,re,glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.simplefilter('ignore')


#(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
import subprocess
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


def mask(_latlon, mask_path):
  """
  array: [lon0,lon1,lat0,lat1]
  """
  lon0,lon1,lat0,lat1 = _latlon

  projection=ccrs.PlateCarree()
  color_polygon = "k"
  lakes_ = cfea.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor=color_polygon, facecolor="none", linewidth=0.5) #facecolor="none"
  
  states_ = cfea.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '10m', edgecolor=color_polygon, facecolor="none", linewidth=0.2)
  lands_ = cfea.NaturalEarthFeature('physical', 'land', '10m')
  # ocean_ = cfea.NaturalEarthFeature('physical', 'ocean', '10m')
  
  """
  外部サイト
  """
  # https://cloud6.net/so/python/3050836
  # https://www.naturalearthdata.com/features/
  
  
  # cfea.COLORS['land']
  # ax.add_feature(cfeature.OCEAN, color='aqua') # 海を水色で塗り潰す
  # ax.add_feature(cfeature.LAKES, color='aqua') # 湖を水色で塗り潰す
  
  f = plt.figure(figsize=(10,10))
  ax = f.add_subplot(1,1,1,projection=projection)
  ax.add_feature(lands_, color='gray')
  
  grd = ax.gridlines(crs=projection,linewidth=1, linestyle=':', color='gray', alpha=0.8)
  # grd.xlocator = mticker.FixedLocator(list(np.arange(lon0,lon1+0.1,1)))
  # grd.ylocator = mticker.FixedLocator(list(np.arange(lat0,lat1+0.1,1)))

  ax.set_extent((lon0,lon1,lat0,lat1),projection)
  ax.coastlines(resolution="10m")
  # ax.add_feature(lakes_)
  ax.add_feature(states_)
  ax.set_title(f"hokuriku_mask", loc="right",pad=None)
  # ax.add_feature(ocean_, color='white')
  # plt.savefig(f"/home/ysorimachi/work/hokuriku/out/snow/mask/hokuriku.png", bbox_inches="tight")
  plt.savefig(f"{mask_path}", bbox_inches="tight")
  # np.savefig(f"{mask_path}.npy", )
  # plt.close()
  return 