# -*- coding: utf-8 -*-
# when   : 2020.0x.xx
# who : [sori-machi]
# what : [ ]
#---------------------------------------------------------------------------
# basic-module
if 1:
  import os, sys, gc
  import glob
  # import datetime as
  from datetime import datetime, timedelta
  import time
  import itertools
  import importlib
  import pickle
  import warnings
  warnings.simplefilter("ignore")

  import matplotlib
  import matplotlib.ticker as mticker
  from mpl_toolkits.axes_grid1 import make_axes_locatable
  # matplotlib.use('Qt5Agg')
  import matplotlib.pyplot as plt
  import matplotlib.cm as cm

  from PIL import Image
  from io import BytesIO
  # for p in sys.path:
  # sys.path.append("/home/ysorimachi/.conda/envs/sori_conda/bin")
  # sys.path.append("/opt/pyenv/versions/miniconda3-latest/envs/anaconda201910/bin")
  # print(sys.path)
  # sys.exit()
  import plotly
  import plotly.graph_objects as go

  import numpy as np
  import pandas as pd
  from pandas import DataFrame, Timestamp, Timedelta
  from pandas.tseries.offsets import Hour, Minute, Second

  try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfea
    import cartopy.io.shapereader as shapereader
    from cartopy.feature import ShapelyFeature
    # initial setting...
    color_polygon = "k"
    lakes_ = cfea.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor=color_polygon, facecolor="none", linewidth=0.5)
    states_ = cfea.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '10m', edgecolor=color_polygon, facecolor='none', linewidth=0.2)
  except:
    print("cartoy not install...")


def plot_map_depth(params,png_path):
    lon0 = params["setmap"][0]
    lon1 = params["setmap"][1]
    lat0 = params["setmap"][2]
    lat1 = params["setmap"][3]
    cmap= params["cmap"]
    _png_path = params["png_"]
    _title = params["title_"]
    # gspan= params["gspan"]
    #cartopy setting........

    projection = ccrs.PlateCarree()
    f = plt.figure(figsize=(20,30))
    # ax = f.add_subplot(1,1,1,projection=projection)
    # f,ax = plt.subplots(2,2,figsize=(2,2),projection=projection)
    # ax.add_geometries(shapes, projection, edgecolor='g', facecolor='g', alpha=0.3)
    for i, input_path in enumerate(_png_path):
      title=_title[i]
      img = np.array(Image.open(input_path))
      dx = 30./ img.shape[1]
      dy = 30./ img.shape[0]
      print("convert change")

      ix0 = int((lon0-120)/dx) 
      ix1 = int((lon1-120)/dx)

      iy0 = int((lat0-20)/dy) 
      iy1 = int((lat1-20)/dy)

      df = pd.DataFrame(img)
      img2 = np.flipud(df.replace(255,np.nan))
      img2 = img2[iy0:iy1,ix0:ix1]

      ax = f.add_subplot(1,len(_title),i+1,projection=projection)
      grd = ax.gridlines(crs=projection)
      grd.xlocator = mticker.FixedLocator(list(np.arange(lon0,lon1+0.1,1)))
      grd.ylocator = mticker.FixedLocator(list(np.arange(lat0,lat1+0.1,1)))

      ax.set_extent((lon0,lon1,lat0,lat1),projection)
      ax.add_feature(lakes_)
      ax.add_feature(states_)
      
      map0 = ax.imshow(img2, extent=(lon0,lon1,lat0,lat1), transform=projection, interpolation='nearest', cmap="jet", vmin=1, vmax=100)
      ax.coastlines(resolution='10m', color=color_polygon, linewidth=0.5)
      # ax.set_extent(_draw_range)
      ax.set_extent((lon0,lon1,lat0,lat1),projection)
      plt.colorbar(map0, pad=0.05, fraction=0.05)
            # path_map = path_outdir + "mapping_{method}_N{n:0=2}_mesh{mesh_size:0=2}km.png".format(method=method,n=n_clusters,mesh_size=mesh_size)
      ax.set_title(title, loc="left",pad=None)
      ax.margins(x=0, y=0)
      # print(f"end {title}...")
    plt.subplots_adjust(wspace=0.1, hspace=0.0)
    plt.tight_layout()
    plt.savefig(png_path, pad_inches=0, bbox_inches='tight')
    plt.close()
    return

if __name__ =="__main__":
  DAT1="/home/ysorimachi/work/snowdepth/dat/tmp"
  OUT1="/home/ysorimachi/work/snowdepth/dat/tmp/png"
  ini_u ="202011121830"
  _utc = pd.date_range(start=ini_u, periods=156, freq="30T")
  _utc = { i+1:ft.strftime("%Y%m%d%H%M") for i,ft in enumerate(_utc) if i%2==1}

  for idx, utc in _utc.items():
    # print(idx,utc)
    cft = str(idx).zfill(3)
    in1=f"{DAT1}/121211/121211-0000SP-0000-{utc}00.png"
    in2=f"{DAT1}/121212_unyo/121212_0000SP_18_{cft}_{ini_u[:10]}0000.png"
    in3=f"{DAT1}/121212_sato/121212_0000SP_18_{cft}_{ini_u[:10]}0000.png"

    params= {
      "setmap":[138.5,146.5,39.9,45.9],
      "cmap": "jet",
      "png_":[in1,in2,in3],
      "title_":["121211","121212_unyo","121212_sato"],
    }
    png_path = f"{OUT1}/plot_image_utc_{utc}.png"
    plot_map_depth(params,png_path)
    # sys.exit()
    print(f"end {idx}- {utc}...")