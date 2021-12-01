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
# import seaborn as sns
#---------------------------------------------------
# sori -module
sys.path.append('/home/griduser/tool')
# from getErrorValues import me,rmse,mae,r2 #(x,y)
# from getPlot import drawPlot,autoLabel #(x=False,y,path),(rects, ax)
# from convSokuhouData import conv2allwithTime,conv2CutCols #(df),(df, ave=minutes,hour)
# from convAmedasData import conv2allwithTime,conv2CutCols #(df),(df, ave=minutes,hour)
# from checkFileExistSize import checkFile #(input_path)
# from plotMonthData import plotDayfor1Month #(df,_col,title=False)
#---------------------------------------------------
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
# initial
#

from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import maskoceans
# from matplotlib.mlab import griddata
import matplotlib.cm as cm

from PIL import Image
from io import BytesIO



llcrnrlat=32.5
llcrnrlon=130.5
urcrnrlat=36.5
urcrnrlon=135

# llcrnrlat=20
# llcrnrlon=120
# urcrnrlat=50
# urcrnrlon=150

# print(iy0,iy1, ix0,ix1)
# sys.exit()
# img2 = np.zeros(shape=(7200, 4800))
# print(img2.shape)
# print(img2[:3,:3])
# sys.exit()

def cutting_area(area):
  if area=="00":  # all area
    return [20, 120, 50, 150]
  elif area=="40":
    return [35, 135, 39, 139]
  elif area=="50": # tyugoku
    return [32.5, 130.5, 36.5 , 135]
  else:
    return [20, 120, 50, 150]


def plot_mesh(in_path, out_path,area="00",iofs =0,iscf=1,vmin=0,vmax=4000,flip=False):
  # img2 = np.zeros(shape=(ny, nx))
  # for ini_j in _init:
  f,ax = plt.subplots(figsize=(10,10))
  llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon = cutting_area(area)

  # img = Image.open(input_path)
  print("convert change")
  # img = np.asarray(Image.open(input_path))
  img = np.array(Image.open(in_path))
  nx = img.shape[1]
  ny = img.shape[0]

  dx = 30./ nx
  dy = 30./ ny

  print("convert change")

  ix0 = int((llcrnrlon-120)/dx) 
  ix1 = int((urcrnrlon-120)/dx)

  iy0 = int((llcrnrlat-20)/dy) 
  iy1 = int((urcrnrlat-20)/dy)

  # print(ix0,ix1,iy0,iy1)
  # sys.exit()
  print("start read ascii file...")

  df = pd.DataFrame(img)
  # df = df.replace(22220,0) # 2020.08.27 background blue
  df = df.replace(22220,np.nan) # 2020.08.28 background white
  df = (df - iofs) * iscf
  _img = df.values
  _lon, _lat = np.meshgrid(np.arange(120,150,dx),np.arange(20,50,dy))

  # print(_lon.shape)
  # print(_lat[:3,:3])
  # print(_img.shape)
  # # sys.exit()
  # # print(rdat(2278, 3716))
  # print(ix0,ix1,iy0,iy1)
  # print("_img[2278,3716]")
  # print(_img[3716,2278])
  # print(_lon[3716,2278])
  # print(_lat[3716,2278])
  # sys.exit()
  # print(_img[-1,-1])
  # print(_lon[-1,-1])
  # print(_lat[-1,-1])
  # sys.exit()
  # sys.exit()


  # llcrnrlat=32.5
  # llcrnrlon=130.5
  # urcrnrlat=36.5
  # urcrnrlon=135

  # df = pd.read_csv(path, delim_whitespace=True, names=_col)
  print("end reading / start ploting mesh...")
  m = Basemap(
      projection='merc',
      llcrnrlat=llcrnrlat,
      llcrnrlon=llcrnrlon,
      urcrnrlat=urcrnrlat,
      urcrnrlon=urcrnrlon,
      ax=ax,
      resolution="h"
  )
  m.drawcoastlines(linewidth=2,color='k')
  m.drawparallels(np.arange(20,50+1,5),labels=[1,0,0,0],linewidth=0.5,)
  m.drawmeridians(np.arange(120,150+1,5),labels=[0,0,0,1],linewidth=0.5)


  # print(_img[3360,2080])
  # print(_lon[3360,2080])
  # print(_lat[3360,2080])

  #cutting
  _lon,_lat = m(_lon[iy0:iy1,ix0:ix1],_lat[iy0:iy1,ix0:ix1])
  # _lat = _lat[iy0:iy1,ix0:ix1]
  # _img2= np.array(()) 
  # for jx in range(ix0,ix1+1): #x
  #   for jy in range(iy0,iy1+1): #
  #     img2[ny-1-jy,jx] = _img[jy,jx]
  #     img2[jy,jx] = _img[ny-1-jy,jx]
  # _img2 = img2[iy0:iy1,ix0:ix1]

  #jwa仕様の為、上下反転を行い使用する
  if flip:
    _img2 = np.flipud(_img)
  else:
    _img2 = _img

  _img2 = _img2[iy0:iy1,ix0:ix1]
  # img = pd.DataFrame(_img)
  # img.to_csv("/home/griduser/work/sola8Now_calc_200826/out/png_mesh/tmp.csv", index=False)
  # print(_lon.shape)
  # print(_lat.shape)
  # print(_img.shape)


  cf = m.contourf(
      _lon, _lat, _img2,
      levels = np.append(np.linspace(vmin,vmax+1,255),9999),
      vmin=vmin,
      vmax=vmax,
      cmap=cm.jet)
  cb = m.colorbar(cf,ticks=range(vmin,vmax+1,250), location = "right", size="2.5%")

  # ax.set_xlabel("lon",fontsize=10,labelpad=10)
  # ax.set_ylabel("lat",fontsize=10,labelpad=10)
  # # fig.suptitle(ini0_u+" _ ft: "+cft)
  # if title:
  #   ax.set_title(month)
  f.savefig(out_path, bbox_inches='tight')
  print("save fig...")
  return


if __name__ == "__main__":

  input_path="/home/griduser/work/sola8Now_calc_200826/tmp/out00.png"
  out_path="/home/griduser/work/sola8Now_calc_200826/tmp/out_rad.png"

  input_path="/work/griduser/tmp/ysorimachi/snowCalc0730/map/mz.png"
  out_path="/work/griduser/tmp/ysorimachi/snowCalc0730/map/height_map.png"


  # plot_mesh(input_path,out_path,area="00",iofs =0,iscf=0.1,flip=True) #nissya
  plot_mesh(input_path,out_path,area="00",iofs =0,iscf=1,vmin=0,vmax=4000,flip=False)
  # plot_mesh(input_path,out_path,title="all_area_map", area="00")
  sys.exit()