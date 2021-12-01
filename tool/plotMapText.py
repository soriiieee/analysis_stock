# -*- coding: utf-8 -*-
# when   : 2020.0x.xx
# who : [sori-machi]
# what : [ ]
#---------------------------------------------------------------------------
# basic-module
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.simplefilter('ignore')
from tqdm import tqdm
import seaborn as sns
#---------------------------------------------------
# sori -module
sys.path.append('/home/griduser/tool')
from getErrorValues import me,rmse,mae,r2 #(x,y)
from getPlot import drawPlot,autoLabel #(x=False,y,path),(rects, ax)
from convSokuhouData import conv2allwithTime,conv2CutCols #(df),(df, ave=minutes,hour)
from convAmedasData import conv2allwithTime,conv2CutCols #(df),(df, ave=minutes,hour)
from checkFileExistSize import checkFile #(input_path)
from plotMonthData import plotDayfor1Month #(df,_col,title=False)
#---------------------------------------------------
# import logging
# from logging import getLogger
# formatter = '%(asctime)s [%(levelname)s]: %(message)s'
# FILE = os.path.basename(sys.argv[0])
# logging.basicConfig(
#  filename='../../log/{}.log'.format(FILE),
#  level=logging.DEBUG,
#  format=formatter)
# logger = getLogger()
# logger.info('-------------m23_02 start------------------')
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
# initial
#
# sys.exit()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap
from matplotlib.collections import PolyCollection
import numpy as np

gspan = 5

lw = 0.5
glw = 0.3
ms = 3
ticsize = 16
labsize = 18
titsize = 20

def station_map_text(_lon, _lat, _text, set_lonlat=[20,50,120,150], figsize=(12,12)):

  f,ax = plt.subplots(figsize=figsize)

  latmin = set_lonlat[0]
  latmax = set_lonlat[1]
  lonmin = set_lonlat[2]
  lonmax = set_lonlat[3]

# sys.exit()

  m = Basemap(resolution='l',llcrnrlat=latmin,llcrnrlon=lonmin,urcrnrlat=latmax,urcrnrlon=lonmax,ax=ax)
  m.drawcoastlines(linewidth=lw,color='k')
  m.drawparallels(np.arange(latmin,latmax+1,gspan),linewidth=glw)
  m.drawmeridians(np.arange(lonmin,lonmax+1,gspan),linewidth=glw)


  # plt.scatter(x,y,s=_z,c=_z, cmap='Blues',vmin=50, vmax=110)
  # plt.text(min_x,min_y+0.1,string)
  # plt.plot(min_x,min_y,marker='o', markersize=20,color="r")
  # plt.colorbar()
  # _lon, _lat = m(_lon,_lat)

  # m.plot(_lon,_lat,c=_z)
  sc = m.scatter(_lon,_lat)
  # cbar = f.colorbar(sc)
  # cb = m.colorbar(location='right', size='3%')
  for lat, lon,txt in zip(_lat, _lon, _text):
    x,y = m(lon,lat)
    # text = f"{name}_{z}"
    plt.text(x, y,txt)
  return ax

if __name__ == "__main__":
  # input_path ="/home/griduser/work/sola8now_200507/Anal/out/png/0806_00/0806_01.csv"
  dir_home="/work/griduser/tmp/ysorimachi/snowCalc0730/map"
  input_path =f"{dir_home}/point_z_all.dat"
  df = pd.read_csv(input_path, delim_whitespace=True, header=None, names=["lon","lat","z"])
  # print(df.head())
  # sys.exit()

  params={
    "setmap":[30,50,130,150],
    "lon": df["lon"].values.tolist(),
    "lat": df["lat"].values.tolist(),
    "z":df["z"].values.tolist(),
    # "name": df["name"].values.tolist(),
    "vminmax":[0,4000]
  }
  f,ax = plt.subplots(figsize=(22,15))
  ax = station_map(ax,params)

  plt.savefig(f"{dir_home}/0908_map_height.png", bbox_inches="tight")

  sys.exit()

