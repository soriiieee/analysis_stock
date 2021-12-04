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
# from getErrorValues import me,rmse,mae,r2 #(x,y)
# from getPlot import drawPlot,autoLabel #(x=False,y,path),(rects, ax)
# from convSokuhouData import conv2allwithTime,conv2CutCols #(df),(df, ave=minutes,hour)
# from convAmedasData import conv2allwithTime,conv2CutCols #(df),(df, ave=minutes,hour)
# from checkFileExistSize import checkFile #(input_path)
# from plotMonthData import plotDayfor1Month #(df,_col,title=False)
# #---------------------------------------------------
# import subprocess
# outd ='/home/griduser/work/sori-py2/deep/out/0924_01'
# os.makedirs(outd, exist_ok=True)
# subprocess.run('rm -f *.png', cwd=outd, shell=True)
# #---------------------------------------------------------------------------
# #---------------------------------------------------------------------------
# # initial
#

import plotly
import plotly.express as px
import plotly.graph_objects as go
# import geopandas as gpd

# geo_df = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))
api_token = open("/home/ysorimachi/env/api_mapbox.env").read()
px.set_mapbox_access_token(api_token)

def map_lonlat_multi(_df,_text,html_path,size=4,zoom=4):
  """
  2021.02.19 edit
  2021.04.14 update
  2021.09.09 update
  """
  lon_center = np.mean(_df[0]["lon"])
  lat_center = np.mean(_df[0]["lat"])
  
  for i,(df,text) in enumerate(zip(_df,_text)):
    df["size"] = size*(i+1)
    fig = px.scatter_mapbox(
      df, lat="lat", lon="lon",size="size",
      color="color",
      # , size="car_hours",
                      # color_continuous_scale=px.colors.cyclical.IceFire, 
                      # 
                      size_max=size)
                      # hover_name=text)
  
  fig.update_layout(
    hovermode='closest',
    mapbox=dict(
        accesstoken=api_token,
        bearing=0,
        center=go.layout.mapbox.Center(
            lat=lat_center,
            lon=lon_center
        ),
        pitch=0,
        zoom=zoom
    )
  )
  print("save fig ->")
  print(html_path)
  # plotly.offline.plot(fig, html_path)  #ファイル
  fig.write_html(html_path)  #ファイル
  return 


def load_AMeDAS(cate=["官"]):
  tbl_path = "../../tbl/ame_master3.csv"
  names = ['ff', 'code', 'cate', 'name', 'name2', 'address', 'lat1', 'lat2','lon1', 'lon2', 'z', 'z_wind', 'z_temp', 'sta','dm1', 'dm2','lon', 'lat', 'dm3', 'x_2400', 'y_3600','dm4','x_480', 'y_720']
  use_col = ['code', 'cate','name', 'name2', 'lon', 'lat']
  df = pd.read_csv(tbl_path)
  df.columns = names
  df = df[use_col]
  df=df.loc[df["cate"].isin(cate),:].reset_index(drop=True)
  df["code_name"] = df[["name","code"]].apply(lambda x:f"{x[0]}({str(x[1])})",axis=1)
  return df


def load_teleme():
  path = "../../tbl/teleme/list_20210115.csv"
  df = pd.read_csv(path)
  df["code_name"] = df["code"].apply(lambda x:f"teleme({str(x)})")
  use_col=["code_name","lon","lat"]

  df = df[use_col]
  return df

def load_rad_point():
  path ="../../tbl/list_obs_sites.csv"
  df = pd.read_csv(path, header=None, names=["code_name","lat","lon"])
  use_point = [ c for c in list(df["code_name"].values)  if "unyo" in c]
  df = df[df["code_name"].isin(use_point)].reset_index(drop=True)
  # df["code_name"] = df["code"].apply(lambda x:f"teleme({str(x)})")
  # use_col=["code_name","lon","lat"]
  return df

def plotly_add_map(_df,text="name",size=5):
  """
  2021.04.10 update
  """
  fig =go.Figure()
  for df in _df:
    fig.add_trace(go.Scattermapbox(
            lat=list(df["lat"].values),
            lon=list(df["lon"].values),
            mode='markers',
            marker=go.scattermapbox.Marker(size=size),
            text=list(df[text].values),))
  return fig

def map_lonlat(df,html_path,text="name",size=4,size_col=None,zoom=4, cmap="Jet"):
  """
  2021.02.19 edit
  2021.04.14 update
  """
  lon_center = np.mean(df["lon"])
  lat_center = np.mean(df["lat"])
  if size_col:
    df["size"] = df[size_col]
    df["size2"] = 4
    size_max = size
  else:
    df["size"] = size
    df["size2"] = size
    size_max = size
  
  if cmap=="jet":
    cmap = px.colors.sequential.Jet
  elif cmap=="plotly":
    cmap = px.colors.sequential.Plotly3
  else:
    # cmap = px.colors.qualitative.T10
    cmap = px.colors.qualitative.Set1
    
  #----
  # 2021.05.13 color scale
  # https://plotly.com/python/builtin-colorscales/
  fig = px.scatter_mapbox(
    df, lat="lat", lon="lon",size="size2",color ="size",
                          # color="peak_hour", size="car_hours",
                    color_continuous_scale=cmap, 
                    # 
                    size_max=size_max,
                    hover_name=text)
  
  fig.update_layout(
    hovermode='closest',
    mapbox=dict(
        accesstoken=api_token,
        bearing=0,
        center=go.layout.mapbox.Center(
            lat=lat_center,
            lon=lon_center
        ),
        pitch=0,
        zoom=zoom
    )
  )
  fig.write_html(html_path)  #ファイル
  return

def map_lonlat2(_df,text,html_path,size=4,zoom=4):
  """
  2021.02.19 edit
  2021.04.14 update
  """
  
  if len(_df)>1:
    fig = plotly_add_map(_df,text=text,size=5)
    fig.write_html(html_path)
    return
  
  lon_center = np.mean(_df[0]["lon"])
  lat_center = np.mean(_df[0]["lat"])
  
  for i,df in enumerate(_df):
    df["size"] = size*(i+1)
    fig = px.scatter_mapbox(
      df, lat="lat", lon="lon",size="size",
      color="color",
      # , size="car_hours",
                      # color_continuous_scale=px.colors.cyclical.IceFire, 
                      # 
                      size_max=size,
                      hover_name=text)
  
  fig.update_layout(
    hovermode='closest',
    mapbox=dict(
        accesstoken=api_token,
        bearing=0,
        center=go.layout.mapbox.Center(
            lat=lat_center,
            lon=lon_center
        ),
        pitch=0,
        zoom=zoom
    )
  )
  
  fig.write_html(html_path)
  return fig

def main(html_name):
  #----
  # tbl_path = "../../tbl/list_nedo_snow.tbl"
  # names = ["code","lon","lat"]
  #----
  ame = load_AMeDAS()
  teleme = load_teleme()
  rad_u = load_rad_point()

  # print(rad_u.head())
  # sys.exit()

  #1 ----------------------------------
  # fig = px.scatter_mapbox(df,lat=df["lat"],lon=df["lon"],zoom=4)

  #2 ----------------------------------
  # for df in [ame,teleme,rad_u]:
  fig = plotly_add_map(_df=[ame,teleme,rad_u]) #def 

  fig.update_layout(
    autosize=True,
    hovermode='closest',
    mapbox=dict(
        accesstoken=api_token,
        bearing=0,
        # center=go.layout.mapbox.Center(lat=36,lon=139),
        center=dict(lat=36,lon=139),
        pitch=0,zoom=6))

  plotly.offline.plot(fig, filename=f'../../out/map/{html_name}.html')  # ファイル
  sys.exit()


def map_lonlat3(df,html_path,zoom=4,size_max=None):
  """
  2021.06.09 update
  2021.06.20 update 3color scales
  needs columns
  :lat
  :lon
  :text
  :size
  """
  
  lat_center = np.mean(df["lat"])
  lon_center = np.mean(df["lon"])
  df["point_size"] = 20
  
  if not size_max:
    size_max=8
    
  # color_continuous_scale= px.colors.sequential.Jet
  color_continuous_scale= px.colors.sequential.Plotly3
  #----
  # 2021.05.13 color scale
  # https://plotly.com/python/builtin-colorscales/
  if "point_size" in df.columns:
    #plotsizeについても変更したい場合
    fig = px.scatter_mapbox(
      df, lat="lat", lon="lon",color="size",size="point_size",
                      color_continuous_scale=color_continuous_scale, 
                      size_max=size_max,
                      hover_name="text")
  else:
    #通常に普通にplotしたい場合
    fig = px.scatter_mapbox(
      df, lat="lat", lon="lon",color="size",
                      color_continuous_scale=color_continuous_scale, 
                      # size_max=20,
                      hover_name="text",range_color=[0,10])
    
  fig.update_layout(
    hovermode='closest',
    mapbox=dict(
        accesstoken=api_token,
        bearing=0,
        center=go.layout.mapbox.Center(
            lat=lat_center,
            lon=lon_center
        ),
        pitch=0,
        zoom=zoom
    )
  )
  # fig.update_yaxes(range=[0,10])
  plotly.offline.plot(fig, filename=html_path) 
  return



if __name__ =="__main__":
  html_name="map_teleme_with_ame_unyo"
  main(html_name)
# px.set_mapbox_access_token(open(".mapbox_token").read())
# fig = px.scatter_mapbox(geo_df,
#                         lat=geo_df.geometry.y,
#                         lon=geo_df.geometry.x,
#                         hover_name="name",
#                         zoom=1)
# fig.show()