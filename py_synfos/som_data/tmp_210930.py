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
from getErrorValues import me,rmse,mae,r2 #(x,y)
from convSokuhouData import conv_sfc #(df, ave=minutes,hour)
#amedas relate 2020.02,04 making...
from tool_AMeDaS import code2name, name2code
from tool_110570 import get_110570,open_110570
from tool_100571 import get_100571,open_100571
#(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
import subprocess
from PIL import Image

from plotMesh import mesh
# mesh(data,lons,lats,title,vmin=0,vmax=1,cmap="seismic",return_ax=False)

DHOME="/work2/ysorimachi/mix_som/dat" #synfos/ data mesh
SYNFOS_INIT = "../../src/now_jst_som.txt"

def load_img(cate):
  
  dl = 100 # 5*dl km 
  path = f"/work2/ysorimachi/mix_som/dat/wrf_0{cate}_12_27.png"
  path = f"{DHOME}/wrf_0{cate}_12_00.png"
  img = np.array(Image.open(path))
  # img = np.where(img ==65535,np.nan,img)
  img = np.where(img>=0,np.nan,img)
  ny,nx = img.shape
  _lat = np.linspace(20,50,ny)
  _lon = np.linspace(120,150,nx)
  
  print(np.nanmin(img), np.nanmax(img))
  img = np.flipud(img)
  
  df = load_10()
  #------- mask filter(個別地点ごと) ---------#
  # for i, r in df.iterrows():
  #   # lon,lat = r[1],r[2]
  #   iy = np.argmin(np.abs(_lat - r[2]))
  #   ix = np.argmin(np.abs(_lon - r[1]))
    
  #   # img[iy-dl:iy+dl,ix-dl:ix+dl] = np.nanmax(img)
  #   img[iy-dl:iy+dl,ix-dl:ix+dl] = 70
  
  iy = np.argmin(np.abs(_lat - 35))
  ix = np.argmin(np.abs(_lon - 135))
  
  dy =int(dl*1.2)
  dx =int(dl*0.8)
  # print(dy,dx,iy,ix, img.shape)
  # sys.exit() 
  img[iy-dy:iy+dy,ix-dx:ix+dx] = 70
  
  #--------------------
  return img,_lon,_lat

def load_10():
  # path = "../../tbl/list_10.tbl"
  path = "/home/ysorimachi/work/ecmwf/tbl/list_10.tbl"
  df = pd.read_csv(path , delim_whitespace=True,header=None)
  df=df[[0,1,2,27]]
  # df = df.set_index(0)
  # name_dict = df.to_dict()[27]
  # return name_dict
  return df
  


def mesh_check(cate):
  OUTD= "/home/ysorimachi/data/synfos/tmp/som_data/mesh"
  cate = "70RH"
  img,_lon,_lat = load_img(cate)

  f ,ax= mesh(img,_lon,_lat,title=cate,vmin=0,vmax=None,cmap="seismic",return_ax=True)

  df = load_10()
  # ax.scatter(df[1],df[2])
  f.savefig(f"{OUTD}/{cate}.png", bbox_inches="tight")
  return


def main():
  if 0:
    cate="70RH"
    mesh_check(cate)
  
  if 1:
    
  return





if __name__ =="__main__":
  if 
  main("07RH")