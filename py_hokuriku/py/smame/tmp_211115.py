# -*- coding: utf-8 -*-
# when   : 2020.0x.xx
# who : [sori-machi]
# what : [ ]
#---------------------------------------------------------------------------
# basic-module
import matplotlib.pyplot as plt
import sys,os,re,glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.simplefilter('ignore')
from tqdm import tqdm
import seaborn as sns
#---------------------------------------------------
# sori -module
sys.path.append('/home/ysorimachi/tool')
sys.path.append('/home/ysorimachi/work/hokuriku/py')
from utils import *


from getErrorValues import me,rmse,mae,r2 #(x,y)
from convSokuhouData import conv_sfc #(df, ave=minutes,hour)
#amedas relate 2020.02,04 making...
from tool_AMeDaS import code2name, name2code
from tool_110570 import get_110570,open_110570
from tool_100571 import get_100571,open_100571
#(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
import subprocess
# outd ='/home/griduser/work/sori-py2/deep/out/0924_01'
# os.makedirs(outd, exist_ok=True)
# subprocess.run('rm -f *.png', cwd=outd, shell=True)

from details_smame2 import pv_mesh
from PIL import Image
from plotMesh import mesh#(data,lons,lats,title,vmin=0,vmax=1,cmap="seismic",return_ax=False)
sys.path.append("..")
from teleme.utils_teleme import load_hh_abc


# local = "/work/ysorimachi/hokuriku/dat2/snow_sfc/origin"
local = "/work/ysorimachi/hokuriku/dat2/rad/tmp" #"2021.11.15"
PWD="/home/ysorimachi/work/hokuriku/py/smame"
BIN="/home/ysorimachi/work/hokuriku/bin"


def download_sfc():
  _name = ["富山","福井"]
  _scode = [ acode2scode(acode=name2code(name)) for name in _name]
  _month = loop_month(st = "201904", ed="202104")
  
  for scode,name in zip(_scode,_name):
    for month in _month:
      if 0: #ftp get
        subprocess.run("sh ../sfc_get2.sh {} {} {}".format(month,scode,local), cwd=PWD,shell=True)
      
      path = f"{local}/sfc_10minh_{month}_{scode}.csv"
      df = pd.read_csv(path)
      df = conv_sfc(df,ave=30)
      print(df.head())
      sys.exit()
      print(datetime.now(), "[end]" , scode, name , month)
      # sys.exit()
  
  return

def triming(img,lonlat=[125,145,25,45]):
  ny,nx = img.shape
  _lon = np.linspace(120,150,nx)
  _lat = np.linspace(20,50,ny)

  x0,x1,y0,y1 = lonlat

  iy0 = np.argmin(np.abs(_lat - y0))
  iy1 = np.argmin(np.abs(_lat - y1))
  ix0 = np.argmin(np.abs(_lon - x0))
  ix1 = np.argmin(np.abs(_lon - x1))
  
  img = img[iy0:iy1,ix0:ix1]
  _lon = _lon[ix0:ix1]
  _lat = _lat[iy0:iy1]
  
  # img = Image.fromarray(img)
  # img = img.resize((size,size))
  # img = np.array(img)
  # img = mms.fit_transform(img)
  return img,_lon,_lat



def plot_png(ini_j="202106061200"):
  OUTD="/home/ysorimachi/work/hokuriku/out/smame/detail/ppt"
  
  def clensing_126051(img):
    img = np.where(img==22220,np.nan, img)
    img = np.flipud(img)
    img /= 10 
    return img
  
  if 0:
    subprocess.run(f"sh {BIN}/get_rad.sh {ini_j} {local}", shell=True)
  
  rad_path = glob.glob(f"{local}/*.png")[0]
  img = np.array(Image.open(rad_path))
  img = clensing_126051(img)
  cmap,vmax = "jet",1200
  
  # ------------rad img -> pu img
  img/=1000 #W->kW
  a,b,c = load_hh_abc("1200")
  img = a*img*img + b*img + c
  img = np.where(img<0,0,img)
  cmap,vmax = "Oranges",1
  #--------------------
  mesh_img = pv_mesh("ALL", reset=False)
  # cmap,vmax = "jet", np.nanmax(img)
  cmap,vmax = "jet", 200
  mesh_img = np.where(mesh_img==0,np.nan,mesh_img)
  img = mesh_img
  # print(np.nanmin(mesh),np.nanmax(mesh))
  
  # area PV ----------------
  # img = img * mesh_img #(pu_mesh * PV 設備)
  # img = np.where(img==0,np.nan,img)
  # print(mesh_img.shape, img.shape)
  # sys.exit()
  
  # ----------------------------------------
  # --- plot --- #
  img,_lon,_lat = triming(img,lonlat=[137-5/3,137+5/3,37-5/3,37+5/3])
  # img,_lon,_lat = triming(img,lonlat=[137.13-0.1,137.13+0.1,36.7-0.1,36.7+0.1])
  f = mesh(img,_lon,_lat,title=None,vmin= 0 , vmax = vmax,cmap=cmap,return_ax=False)
  f.savefig(f"{OUTD}/rad_mesh.png",bbox_icnhes="tight")
  
  # ----------------------------------------
  




if __name__ == "__main__":
  
  # scode = acode2scode(acode=name2code("福井"))
  plot_png()