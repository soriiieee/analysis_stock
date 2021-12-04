# -*- coding: utf-8 -*-
# when   : 2020.0x.xx
# who : [sori-machi]
# what : [ ]
#---------------------------------------------------------------------------
# basic-module
if 1:
  import os, sys, gc
  import glob
  import datetime as dt
  import time
  import itertools
  import importlib
  import pickle
  import warnings
  warnings.simplefilter("ignore")

  import matplotlib
  # 自作kara-mapの作成
  from matplotlib.colors import ListedColormap, BoundaryNorm
  # https://hackmd.io/@h2tg95D2RP2ed-D8u-49Mg/S1moBqaRr
  import matplotlib.ticker as mticker
  from mpl_toolkits.axes_grid1 import make_axes_locatable
  # matplotlib.use('Qt5Agg')
  import matplotlib.pyplot as plt
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
  
  from matplotlib.patches import Circle #((), radius=5,fill = False, color = "k" ,lw=1)[source]
  from matplotlib.patches import Arrow #(xy, radius=5, **kwargs)[source]

  try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfea
    # import cartopy.feature as cfea
    import cartopy.io.shapereader as shapereader
    from cartopy.feature import ShapelyFeature
  except:
    print("could not import cartopy")
    # https://www.naturalearthdata.com/download/10m/cultural/ne_10m_admin_1_states_provinces.zip
# matplot で日本語
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 15
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']


def mesh(data,lons,lats,title,vmin=0,vmax=1,cmap="seismic",return_ax=False):
    """
    data: numpy(:,:) :"mesh
    lons: numpy(:,:) or array
    lats: numpy(:,:) or array
    title : valuesname
    
    """
    #cartopy setting........
    lon0,lon1 = np.min(lons),np.max(lons)
    lat0,lat1 = np.min(lats),np.max(lats)
    # print(lon0,lon1,lat0,lat1)
    # sys.exit()
    
    projection = ccrs.PlateCarree()
    f = plt.figure(figsize=(7,7))
    ax = f.add_subplot(111,projection=projection)
    color_polygon = "k"
    lakes_ = cfea.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor=color_polygon, facecolor="none", linewidth=0.5)
    ax.add_feature(lakes_)
    states_ = cfea.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '10m', edgecolor=color_polygon, facecolor='none', linewidth=0.2)
    ax.add_feature(states_)
    
    # grd = ax.gridlines(crs=projection)
    # grd.xlocator = mticker.FixedLocator(list(np.arange(lon0,lon1+0.1,1)))
    # grd.ylocator = mticker.FixedLocator(list(np.arange(lat0,lat1+0.1,1)))

    ax.set_extent((lon0,lon1,lat0,lat1),projection)

    # map1 = ax.imshow(data, extent=(lon0,lon1,lat0,lat1), transform=projection, interpolation='nearest', cmap="cool", vmin=0, vmax=100)
    map1 = ax.imshow(data, extent=(lon0,lon1,lat0,lat1), transform=projection, interpolation='nearest',cmap=cmap,vmin=vmin,vmax=vmax)

    # url = 'https://map1c.vis.earthdata.nasa.gov/wmts-geo/wmts.cgi'
    # layer = 'MODIS_Water_Mask'
    # ax.add_wmts(url, layer)
  
    ax.coastlines(resolution='10m', color=color_polygon, linewidth=0.5)
    # ax.plot(point[0],point[1],marker="o", color="k")
    plt.colorbar(map1, pad=0.05, fraction=0.05, shrink=0.7)
    ax.set_title(title, loc="left",pad=None)
    ax.margins(x=0, y=0)
    plt.tight_layout()
    # plt.savefig(png_path, pad_inches=0, bbox_inches='tight')
    if return_ax:
      return f,ax
    else:
      return f

def mesh2(ax,data,lons,lats,title,vmin=996,vmax=1032,cmap="seismic",return_ax=False):
  """
  data: numpy(:,:) :"mesh
  ax : 呼び出す前にprojectionをかませたaxを表示しておく
    f = plt.figure(figsize=(7,7))
    ax = f.add_subplot(111,projection=projection)
  lons: numpy(:,:) or array
  lats: numpy(:,:) or array
  title : valuesname
  
  """
  #cartopy setting........
  lon0,lon1 = np.min(lons),np.max(lons)
  lat0,lat1 = np.min(lats),np.max(lats)
  # print(lon0,lon1,lat0,lat1)
  # sys.exit()
  # print(type(ax))
  # print(type(plt.axes(projection=ccrs.PlateCarree())))
  # sys.exit()
  
  projection = ccrs.PlateCarree()
  color_polygon = "k"
  lakes_ = cfea.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor=color_polygon, facecolor="none", linewidth=0.5)
  ax.add_feature(lakes_)
  states_ = cfea.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '10m', edgecolor=color_polygon, facecolor='none', linewidth=0.2)
  ax.add_feature(states_)
  
  grd = ax.gridlines(crs=projection)
  # grd.xlocator = mticker.FixedLocator(list(np.arange(lon0,lon1+0.1,1)))
  # grd.ylocator = mticker.FixedLocator(list(np.arange(lat0,lat1+0.1,1)))

  ax.set_extent((lon0,lon1,lat0,lat1),projection)

  # map1 = ax.imshow(data, extent=(lon0,lon1,lat0,lat1), transform=projection, interpolation='nearest', cmap="cool", vmin=0, vmax=100)
  map1 = ax.imshow(data, extent=(lon0,lon1,lat0,lat1), transform=projection, interpolation='nearest',vmin=vmin, vmax=vmax,cmap=cmap)
  # 960,1040
  # ax.add_feature(cfea.OCEAN, zorder=1000, edgecolor='k') #2021.08.31
  # ax.add_feature(cfea.OCEAN, alpha=1, facecolor="white",edgecolor='k') #2021.08.31

  # ax.set_global()
  
  ax.coastlines(resolution='10m', color=color_polygon, linewidth=0.5)
  # ax.plot(point[0],point[1],marker="o", color="k")
  plt.colorbar(map1, pad=0.03, fraction=0.05, shrink=0.7)
  # plt.colorbar(map1, pad=0.03,shrink=0.7)
  ax.set_title(title, loc="left",pad=None, fontsize=8)
  # ax.margins(x=0, y=0)
  plt.tight_layout()
  # plt.savefig(png_path, pad_inches=0, bbox_inches='tight')
  if return_ax:
    return ax
  else:
    return

def mesh3(data,lons,lats,title,png_path):
  """
  data: numpy(:,:) :"mesh
  lons: numpy(:,:) or array
  lats: numpy(:,:) or array
  title : valuesname
  """
  #cartopy setting........
  lon0,lon1 = np.min(lons),np.max(lons)
  lat0,lat1 = np.min(lats),np.max(lats)
  
  # lat0,lat1 = 32.621481,36.621481 
  # lon0,lon1 =136.148811,140.148811
  # print(lon0,lon1,lat0,lat1)
  # sys.exit()
  
  projection = ccrs.PlateCarree()
  f = plt.figure(figsize=(7,7))
  ax = f.add_subplot(111,projection=projection)
  color_polygon = "k"
  lakes_ = cfea.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor=color_polygon, facecolor="none", linewidth=0.5)
  ax.add_feature(lakes_)
  states_ = cfea.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '10m', edgecolor=color_polygon, facecolor='none', linewidth=0.2)
  ax.add_feature(states_)
  
  # grd = ax.gridlines(crs=projection)
  # grd.xlocator = mticker.FixedLocator(list(np.arange(lon0,lon1+0.1,1)))
  # grd.ylocator = mticker.FixedLocator(list(np.arange(lat0,lat1+0.1,1)))

  ax.set_extent((lon0,lon1,lat0,lat1),projection)

  # map1 = ax.imshow(data, extent=(lon0,lon1,lat0,lat1), transform=projection, interpolation='nearest', cmap="cool", vmin=0, vmax=100)
  # map1 = ax.imshow(data, extent=(lon0,lon1,lat0,lat1), transform=projection, interpolation='nearest',vmin=0, vmax=3)
  # map1 = ax.contour(data, extent=(lon0,lon1,lat0,lat1), transform=projection,vmin=1, vmax=2)
  cmap = ListedColormap(['yellow','red'])
  bounds = np.linspace(1,2,3)
  norm = BoundaryNorm(bounds,cmap.N)
  map1 = ax.imshow(data, extent=(lon0,lon1,lat0,lat1), transform=projection,cmap=cmap,norm=norm,vmin=min(bounds),vmax=max(bounds))
  
  ax.coastlines(resolution='10m', color=color_polygon, linewidth=0.5)
  # ax.plot(point[0],point[1],marker="o", color="k")
  plt.colorbar(map1, pad=0.05, fraction=0.05, shrink=0.7)
  #---------
  """
  2021.06.30 竜巻案件
  """
  ax.plot(138.148811,34.62148, marker="o", markersize=5, color="k",transform=projection)
  # tornado_latlon = [34.650355,138.186185]
  tornado_latlon = [34.7400282,138.2246436] 
  dx,dy = .07,.07
  x0,y0 = tornado_latlon[1],tornado_latlon[0] 
  circ = Circle(xy=(138.148811,34.62148), radius=.07,fill=False, lw=1, color="k")
  arrw = Arrow(x0-dx, y0+dy, dx, -dy, width=0.07,color="k")
  ax.add_patch(circ)
  ax.add_patch(arrw)
  try:
    ax.set_title(title, loc="left",pad=None, fontsize=20)
    
    """ 個別事案 """
    # ax.plot(138.148811,34.62148, marker="o", markersize=10, color="r",transform=projection)
  except:
    pass
  ax.margins(x=0, y=0)
  plt.tight_layout()
  plt.savefig(png_path, pad_inches=0, bbox_inches='tight')
  return


def contour2(ax,data,lons,lats,title,vmin=996,vmax=1032,cmap="seismic"):
  
  """
  data: numpy(:,:) :"mesh
  ax : 呼び出す前にprojectionをかませたaxを表示しておく
    f = plt.figure(figsize=(7,7))
    ax = f.add_subplot(111,projection=projection)
  lons: numpy(:,:) or array
  lats: numpy(:,:) or array
  title : valuesname
  
  """
  #cartopy setting........
  lon0,lon1 = np.min(lons),np.max(lons)
  lat0,lat1 = np.min(lats),np.max(lats)
  # print(lon0,lon1,lat0,lat1)
  # sys.exit()
  x,y = np.meshgrid(lons,lats)
  
  projection = ccrs.PlateCarree()
  color_polygon = "k"
  lakes_ = cfea.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor=color_polygon, facecolor="none", linewidth=0.5)
  ax.add_feature(lakes_)
  states_ = cfea.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '10m', edgecolor=color_polygon, facecolor='none', linewidth=0.2)
  ax.add_feature(states_)
  
  # grd = ax.gridlines(crs=projection)
  # grd.xlocator = mticker.FixedLocator(list(np.arange(lon0,lon1+0.1,1)))
  # grd.ylocator = mticker.FixedLocator(list(np.arange(lat0,lat1+0.1,1)))

  ax.set_extent((lon0,lon1,lat0,lat1),projection)

  # map1 = ax.imshow(data, extent=(lon0,lon1,lat0,lat1), transform=projection, interpolation='nearest', cmap="cool", vmin=0, vmax=100)
  # map1 = ax.imshow(data, extent=(lon0,lon1,lat0,lat1), transform=projection, interpolation='nearest')
  
  clevs = np.arange(vmin,vmax,4)
  cs = ax.contour(x, y, data, levels=clevs)
  clevels = cs.levels # ラベルを付ける値
  cs.clabel(clevels, fontsize=12) # 等高線ラベル
  
  ax.coastlines(resolution='10m', color=color_polygon, linewidth=0.5)
  # ax.plot(point[0],point[1],marker="o", color="k")
  # plt.colorbar(cs, pad=0.05, fraction=0.05, shrink=0.7)
  ax.set_title(title, loc="left",pad=None, fontsize=8)
  ax.margins(x=0, y=0)
  plt.tight_layout()
  # plt.savefig(png_path, pad_inches=0, bbox_inches='tight')
  return ax

def mesh_sample_japan(lons,lats, png_path):
  lon0,lon1 = np.min(lons),np.max(lons)
  lat0,lat1 = np.min(lats),np.max(lats)
  
  projection = ccrs.PlateCarree()
  f = plt.figure(figsize=(7,7))
  ax = f.add_subplot(111,projection=projection)
  color_polygon = "k"
  lakes_ = cfea.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor=color_polygon, facecolor="none", linewidth=1,color="k")
  ax.add_feature(lakes_)
  states_ = cfea.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '10m', edgecolor=color_polygon, facecolor='none', linewidth=1,color="k")
  ax.add_feature(states_)
  
  # grd = ax.gridlines(crs=projection)
  # grd.xlocator = mticker.FixedLocator(list(np.arange(lon0,lon1+0.1,1)))
  # grd.ylocator = mticker.FixedLocator(list(np.arange(lat0,lat1+0.1,1)))

  ax.set_extent((lon0,lon1,lat0,lat1),projection)
  # map1 = ax.imshow(data, extent=(lon0,lon1,lat0,lat1), transform=projection, interpolation='nearest', cmap="cool", vmin=0, vmax=100)
  # map1 = ax.imshow(data, extent=(lon0,lon1,lat0,lat1), transform=projection, interpolation='nearest',cmap=cmap,vmin=vmin,vmax=vmax)
  
  ax.coastlines(resolution='10m', color=color_polygon, linewidth=1)
  # ax.plot(point[0],point[1],marker="o", color="k")
  # plt.colorbar(map1, pad=0.05, fraction=0.05, shrink=0.7)
  # ax.set_title(title, loc="left",pad=None, fontsize=8)
  # ax.margins(x=0, y=0)
  # plt.tight_layout()
  plt.savefig(png_path,bbox_inches="tight")
  return plt

def axmap(ax ,area):
  lon0,lon1,lat0,lat1 = area
  projection = ccrs.PlateCarree()
  # f = plt.figure(figsize=(7,7))

  ax.projection = projection 
  # setting---------------------
  color_polygon = "k"
  lakes_ = cfea.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor=color_polygon, facecolor="none", linewidth=1,color="k")
  ax.add_feature(lakes_)
  states_ = cfea.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '10m', edgecolor=color_polygon, facecolor='none', linewidth=1,color="k")
  ax.add_feature(states_)
  # grd = ax.gridlines(crs=projection)
  # grd.xlocator = mticker.FixedLocator(list(np.arange(lon0,lon1+0.1,1)))
  # grd.ylocator = mticker.FixedLocator(list(np.arange(lat0,lat1+0.1,1)))
  ax.set_extent((lon0,lon1,lat0,lat1),projection)
  ax.coastlines(resolution='10m', color=color_polygon, linewidth=1)
  return ax


def plt_multi_map(H,W):
  # projection = ccrs.PlateCarree()
  f,ax = plt.subplots(H,W,figsize=(5*W,5*H), subplot_kw={'projection': ccrs.PlateCarree()})
  return f,ax


if __name__ =="__main__" :
    #director setting........
    OUT3="/home/ysorimachi/work/8now_cast/out/1112_shikoku/8now0/out3"
    # make tbl -------------------------------------------------------
    out_path="/home/ysorimachi/work/8now_cast/tbl/list_s4ku_201116.csv"
    csv_path=f"/home/ysorimachi/work/8now_cast/out/1112_shikoku/8now0/out3/cluster_map.csv"
    # mk_point_list(nx=4800.,ny=7200, out_path=out_path)
    # sys.exit()
    df = pd.read_csv(csv_path)
    _col = [ col for col in df.columns if "wald" in col]

    _array = np.arange(2,47,4)
    # sys.exit()
    for n_sta in _array:
# lonmin, lonmax, latmin, latmax
        params={
            "setmap":tuple([131,135,32,35]),
            "lon": df["lon"].values.tolist(),
            "lat": df["lat"].values.tolist(),
            "z": [ "wald_"+str(n_sta+i) for i in range(4) ],
            # "name": df["name"].values.tolist(),
            # "vminmax":[0,n_cluster],
            "cmap": "Set1"
        }
        png_path = f"{OUT3}/png/{n_sta}.png"
        plot_map_cartopy(df,params,png_path)
        # sys.exit()
        print(f"end {n_sta}...")
        #cartopy.setting............



    sys.exit()
    #-------------------------------------------------------
    tbl = pd.read_csv(out_path)
    out_df=tbl[["name","lon","lat"]]
    # print(tbl.head())
    # sys.exit()
    # make all sites concat to 1file -------------------------------------------------------
    # mk_all_concat()
    # sys.exit()
    datadir="/home/ysorimachi/work/8now_cast/out/1112_shikoku/8now0/out2/sites2"

    input_path="/home/ysorimachi/work/8now_cast/out/1112_shikoku/8now0/out3/all_5km.csv"
    input_path="/home/ysorimachi/work/8now_cast/out/1112_shikoku/8now0/out3/all_5km_seiten.csv"
    df = pd.read_csv(input_path)
    df = df.drop(["dti"],axis=1).T.reset_index()

    tmp =df.iloc[:,1:].dropna(axis=1)
    # print(tmp.head())
    # sys.exit()


    # make all sites concat to 1file -------------------------------------------------------
    _method=["wald"]
    _n_clusters = [ i for i in range(2,51+1)]
    csv_path=f"/home/ysorimachi/work/8now_cast/out/1112_shikoku/8now0/out3/cluster_map.csv"
    calc_cluster_csv(tmp,_n_clusters,_method,out_df,csv_path=csv_path)

    sys.exit()

