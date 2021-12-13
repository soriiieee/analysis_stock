# -*- coding: utf-8 -*-
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
  import tqdm
  warnings.simplefilter("ignore")

  #描画系
  import matplotlib
  import matplotlib.ticker as mticker
  from mpl_toolkits.axes_grid1 import make_axes_locatable
  # matplotlib.use('Qt5Agg')
  import matplotlib.pyplot as plt
  import matplotlib.cm as cm

  #png読み込み
  from PIL import Image
  from io import BytesIO
  import plotly
  import plotly.graph_objects as go

  import numpy as np
  import pandas as pd
  import subprocess
  try:
    # -----------------------------------------------------------------
    # mapping tool (~GMT) --------- 
    import cartopy.crs as ccrs
    import cartopy.feature as cfea
    import cartopy.io.shapereader as shapereader
    from cartopy.feature import ShapelyFeature
    # initial setting...
    color_polygon = "k"
    lakes_ = cfea.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor=color_polygon, facecolor="none", linewidth=0.5)
    states_ = cfea.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '10m', edgecolor=color_polygon, facecolor='none', linewidth=0.2)
    # -----------------------------------------------------------------
  except:
    print("cartoy not install...")
  
  import cv2

  # "/home/ysorimachi/.conda/envs/sori_conda/lib/python3.7/site-packages/cv2.cpython-37m-x86_64-linux-gnu.so"

#画像処理汎用モジュール

"""
静止画からmp4動画を作成する
参考：https://yusei-roadstar.hatenablog.com/entry/2019/11/29/174448

"""
def concat_img(ini_j):
  data_d = "/home/ysorimachi/work/hokuriku/out/snow"
  path0 = f"{data_d}/111095_2020{ini_j}00.png"
  # path21 = f"{data_d}/111095_2021{ini_j}00.png"
  path1 = f"{data_d}/121212_2020{ini_j}00.png"
  img0 = cv2.imread(path0)
  img1 = cv2.imread(path1)
  # print(img0.shape, img1.shape)
  # sys.exit()

  img = np.concatenate([img0, img1], 1)
  return img

def get_paths(cate):
  dhome = "/home/ysorimachi/work/hokuriku/out/snow/store"
  log = pd.read_csv("../../log/log_loop_bk.log",header=None,delim_whitespace=True)
  log[6] = log[6].astype(str)
  _ini_j = log[6].values.tolist()
  
  _paths = [ f"{dhome}/{ini_j}/111095_{ini_j}_{cate}.png" for ini_j in _ini_j]
  return _paths

def mk_movie(_img,movie_path,frame_rate=4):
  fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
  height,width,channel = _img[0].shape
  
  video = cv2.VideoWriter(movie_path,fourcc,frame_rate,(width,height))
  print("動画変換開始")
  for img in _img:
    video.write(img)
  video.release()
  print("動画ファイルのパスは下記になります")
  print(movie_path)
  return

if __name__ =="__main__":
  # setting -----------------------------------
  # cate = "rain" #fct
  cate = "s_smame"
  cate = "a_smame"
  # cate = "s_smame"
  movie_path = f"/home/ysorimachi/work/hokuriku/out/snow/movie/{cate}.mp4"
  mk_movie(cate, movie_path)
  sys.exit()
  
  
  #setting -----------------------------------
  # /home/ysorimachi/work/sori_py2/sori_weather/bin
  # out_d = "/home/ysorimachi/work/hokuriku/tmp"
  #start function...
   #[lat_min ,lon_min,lat_max,lon_max] (google map 等から)
  # movie_path = f"{out_d}/comparison_jwa_2021.mp4"
  # frame_rate=8
  # conv_img2movie(movie_path,2)