# -*- coding: utf-8 -*-
# when   : 2020.0x.xx
# who : [sori-machi]
# what : [ ]
#---------------------------------------------------------------------------
# basic-module
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
# from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import os
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
# from getPlot import drawPlot,autoLabel #(x=False,y,path),(rects, ax)
# from convSokuhouData import conv2allwithTime,conv2CutCols #(df),(df, ave=minutes,hour)
# from convAmedasData import conv2allwithTime,conv2CutCols #(df),(df, ave=minutes,hour)
# from checkFileExistSize import checkFile #(input_path)
# from plotMonthData import plotDayfor1Month #(df,_col,title=False)
#---------------------------------------------------
# import subprocess
# outd ='/home/griduser/work/sori-py2/deep/out/0924_01'
# os.makedirs(outd, exist_ok=True)
# subprocess.run('rm -f *.png', cwd=outd, shell=True)
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
# initial
#
# from mpl_toolkits.basemap import Basemap
# from mpl_toolkits.basemap import shiftgrid
import pygrib
# import cartopy.crs as ccrs
# import cartopy.feature as cfea
# # import cartopy.feature as cfea
# import cartopy.io.shapereader as shapereader
# from cartopy.feature import ShapelyFeature

# ---
# return : _code,_scode,_name,_lon,_lat
from conv_amedas_30min import load_synfos_table


def isint(x):
  try:
    int(x)
    return 1
  except:
    return 0

def  mk_weather_element_list():
  d0 = {i:[] for i in range(8)}
  with open("../tbl/j2e_data_keys.dat") as f:
    for i,f_mini in enumerate(f.readlines()):
      r = f_mini.split("\n")[0].split(":")
      for j,val in enumerate(r):
        d0[j].append(val)
  
  df = pd.DataFrame()
  for j in range(8):
    df[f"e{j}"] =d0[j] 

  print(df.head())
  names = ["n","ele","unit","regular","h","level","ft","ft0"]
  df.columns = names
  df.to_csv("../tbl/list_grib_keys.csv", index=False)
  return

def plot_mesh(_n,n_fig_row,n_fig_col,dict_ele, gribs,png_path):
  # print(gribs.select())
  # sys.exit()
  grb = gribs.select()[0]
  f = plt.figure(figsize=(20*n_fig_col,20*n_fig_row))
  lon0 = float(grb['longitudeOfFirstGridPointInDegrees'])
  lon1 = float(grb['longitudeOfLastGridPointInDegrees'])
  lat1 = float(grb['latitudeOfFirstGridPointInDegrees'])
  lat0 = float(grb['latitudeOfLastGridPointInDegrees'])
  nx = int(grb['Ni'])
  ny = int(grb['Nj'])
  # sys.exit()
  # element = gribs.select()[n]
  # print(element)
  # sys.exit()
  # print(lon0,lon1,lat0,lat1,nx,ny)

  # sys.exit()

  projection=ccrs.PlateCarree()
  color_polygon = "k"
  lakes_ = cfea.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor=color_polygon, facecolor="none", linewidth=0.5)
  states_ = cfea.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '10m', edgecolor=color_polygon, facecolor='none', linewidth=0.2)
  # sys.exit()
  # for i, zcol in enumerate(["1","2","3","4"]):
  for i,n in enumerate(_n):
    ax = f.add_subplot(n_fig_row,n_fig_col,i+1,projection=projection)
    ele_name = dict_ele[n]
    data = gribs.select()[n].values
    max0 = np.round(np.max(data),3)
    min0 = np.round(np.min(data),3)
    mean0 = np.round(np.mean(data),3)

    grd = ax.gridlines(crs=projection)
    grd.xlocator = mticker.FixedLocator(list(np.arange(lon0,lon1+0.1,1)))
    grd.ylocator = mticker.FixedLocator(list(np.arange(lat0,lat1+0.1,1)))

    #plot imshow ----------------------------
    map0 = ax.imshow(data, extent=(lon0,lon1,lat0,lat1), transform=projection, interpolation='nearest', cmap="jet") #vmin,vmax
    # map0 = ax.imshow(data,transform=projection, interpolation='nearest', cmap="jet") #vmin,vmax
    # ax.coastlines(resolution='10m', color=color_polygon, linewidth=0.5)
    # ax.set_extent(_draw_range)
    ax.set_extent((lon0,lon1,lat0,lat1),projection)
    plt.colorbar(map0, pad=0.05, fraction=0.05)
          # path_map = path_outdir + "mapping_{method}_N{n:0=2}_mesh{mesh_size:0=2}km.png".format(method=method,n=n_clusters,mesh_size=mesh_size)
    ax.set_title(f"{n}:{ele_name}/max={max0}/min={min0}/mean={mean0}", loc="left",pad=None)
    # ax.set_extent((lon0,lon1,lat0,lat1),projection)
    # ax.coastlines(resolution="10m")
    ax.add_feature(lakes_)
    ax.add_feature(states_)

  # sys.exit()
  plt.subplots_adjust(wspace=0.1, hspace=0.0)
  plt.savefig(png_path,bbox_inches="tight", pad_inches=0)
  plt.close()
  return

def get_ixiy(nx,ny,lon0,lon1,lat0,lat1,rlon,rlat):
  rdx = (lon1 -lon0) / nx
  rdy = (lat0 -lat1) / ny

  ix = int(np.floor((rlon-lon0)/rdx))
  iy = int(np.floor((lat0-rlat)/rdy))
  return ix,iy


if __name__ =="__main__":
  DATAHOME="/home/ysorimachi/data/ecmwf/dat/D1E"
  cele='Surface solar radiation downwards'

  _code,_scode,_name,_lon,_lat = load_synfos_table()


  ini_u=sys.argv[1]
  fcs_u=sys.argv[2]

  _t = pd.date_range(start = fcs_u, freq="3H",periods = 9)
  _utc8 = [ t.strftime("%m%d%H%M") for t in _t ]

  _val=[]
  _utc=[]
  _code_all=[]
  _isCF=[]
  _ens_n=[]

  for utc8 in _utc8:
    input_path = f"{DATAHOME}/{ini_u[:10]}/D1E{ini_u[4:12]}{utc8}1"

    if os.path.exists(input_path):
      gribs = pygrib.open(input_path)

      lon0 = gribs.select()[0]['longitudeOfFirstGridPointInDegrees']
      lon1 = gribs.select()[0]['longitudeOfLastGridPointInDegrees']
      lat0 = gribs.select()[0]['latitudeOfFirstGridPointInDegrees']
      lat1 = gribs.select()[0]['latitudeOfLastGridPointInDegrees']
      nx = gribs.select()[0]['Ni']
      ny = gribs.select()[0]['Nj']

      # print(nx,ny,lon0,lon1,lat0,lat1)
      # sys.exit()
      i=0
      for grb in gribs:
        # print(grb.keys())
        # sys.exit()
        if grb["parameterName"]==cele:
          isCF = grb["perturbationNumber"]
          for code,rlon,rlat in zip(_code,_lon,_lat):
            # print(code,rlon,rlat)
            # sys.exit()
            ix,iy = get_ixiy(nx,ny,lon0,lon1,lat0,lat1,rlon,rlat)
            # print(grb["distinctLatitudes"][iy])
            # print(code, ix,iy)
            # sys.exit()
            
            try:
              val = grb.values[iy,ix]
            except:
              val= 9999.
            
            #input_array
            # print(val,utc8,code,isCF,i)
            # sys.exit()
            _val.append(val)
            _utc.append(utc8)
            _code_all.append(code)
            _isCF.append(isCF)
            _ens_n.append(i)
          i+=1
        else:
          pass
      # sys.exit()
    print(f"end :{ini_u} - {utc8}")
  
  df = pd.DataFrame()
  df["ft_utc"] = _utc
  df["code"] = _code_all
  df["isCF"] = _isCF
  df["rad0"] = _val
  df["n_ens"] = _ens_n
  df["ini_u"] = ini_u

  df.to_csv(f"/home/ysorimachi/data/ecmwf/out/1126/{ini_u}.csv", index=False)
  # sys.exit()