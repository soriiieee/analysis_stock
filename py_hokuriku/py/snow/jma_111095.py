# -*- coding: utf-8 -*-
# when   : 2021.03.10 init
# when   : 2021.08.31 update
# when   : 2021.11.10 update
# who : [sori-machi]
# what : [ ]
#---------------------------------------------------------------------------
# basic-module
import sys,os,re,glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.simplefilter('ignore')
from tqdm import tqdm
import seaborn as sns


sys.path.append('/home/ysorimachi/tool')
from tool_cmap import cmap_snow
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

#(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
import subprocess
import pygrib
import cartopy.crs as ccrs
import cartopy.feature as cfea
# import cartopy.feature as cfea
import cartopy.io.shapereader as shapereader
from cartopy.feature import ShapelyFeature

from cartopy_land_mask import mask #_latlon=[120,150,20,50], mask_path

from netCDF4 import Dataset
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy

sys.path.append("/home/ysorimachi/work/hokuriku/py")
# from smame.eda_map import smame_position #(cate ="all", dd8="20200330")
# from utils import load_teleme_position
from tool_time import dtinc
from utils_log import log_write
import subprocess

from teleme.utils_teleme import mk_teleme_table, teleme_max,load_teleme#(ONLY_df=False)
from utils_snow import near_rad
sys.path.append("..")
from utils import load_rad
from tool_cmap import cmap_snow


def get_lonlat(name="富山"):
  df=pd.read_csv("../../tbl/ame_master3.csv")
  try:
    lon,lat = list(df.loc[df["観測所名"]==name,["経度","緯度"]].values[0])
  except:
    lon,lat = 9999,9999
  return lon,lat


def save_depth(xx,yy,data,out_path):
  """[summary]

  Args:
      xx ([type]): [llongtude]
      yy ([type]): [latuitude]
      data ([type]): [description]
      out_path　: output-csv
      
  return :
      None(出力すべき先に保存して格納しておく-->teleme) 
  """
  _val,_lon,_lat,_name = [],[],[],[]
  
  df = mk_teleme_table(ONLY_df=True)
  # print(np.nanmin(data), np.nanmax(data))
  # sys.exit()
  
  # for name in _list:
  for i,r in df.iterrows():
    name, lon,lat = r["code"],r["lon"], r["lat"]
    # print(name,lon,lat)
    # sys.exit()
    
    if (130 < lon< 140) and (30<lat<40):
      jx = np.argmin(np.abs(xx -lon))
      jy = np.argmin(np.abs(yy - lat))
      val = data[jy,jx]

      if (0 < val< 500): 
        _val.append(np.round(val,1))
      else:
        _val.append(0)
        
      _lon.append(np.round(lon,3))
      _lat.append(np.round(lat,3))
      _name.append(name)
    else:
      _val.append(9999)
      _lon.append(9999)
      _lat.append(9999)
      _name.append(name)

  df = pd.DataFrame()
  df["name"] = _name
  df["lon"] = _lon
  df["lat"] = _lat
  df["val"] = _val
  # df.to_csv(f"/home/ysorimachi/work/hokuriku/out/snow2/csv/111095_{ini_j}.csv", index=False, encoding="shift-jis")
  df.to_csv(out_path, index=False)
  return


def read_111095(ini_j, cate=None,size=None,vmin=0.1,vmax=500):
  """
  init: 2021.02.20
  update: 2021.03.01 
  update: 2021.11.09 
  [main program]
  convert grib type data to netcdf datatyepes and use "Dataset" python libraly which convert to pandas df for plot map and timeseries datas asnd so on..  after get grib files
  
  """
  # ini_u = os.path.basename(path).split(".")[0]
  # ini_j = conv2utc(ini_u,ctype="J")
  ini_u = dtinc(ini_j,4,-9)
  path = f"/home/ysorimachi/work/hokuriku/dat/snow/111095/{ini_u}.nc"
  nc = Dataset(path,"r")

  # lon0,lon1,lat0,lat1 = 135.5,137.8,35.5,37.8
  # ix0 = np.argmin(np.abs(np.array(nc.variables["longitude"]) - 135.5))
  # ix1 = np.argmin(np.abs(np.array(nc.variables["longitude"]) - 138.0))
  # iy0 = np.argmin(np.abs(np.array(nc.variables["latitude"]) - 35.5))
  # iy1 = np.argmin(np.abs(np.array(nc.variables["latitude"]) - 38.0))

  # xx = nc.variables["longitude"][ix0:ix1]
  # yy = nc.variables["latitude"][iy0:iy1]
  # data = nc.variables["var0_1_232_surface"][0,iy0:iy1,ix0:ix1]*100 #m->cm
  xx = nc.variables["longitude"][:]
  yy = nc.variables["latitude"][:]
  # data = nc.variables["var0_1_232_surface"][0,iy0:iy1,ix0:ix1]*100 #m->cm
  data = nc.variables["var0_1_232_surface"][0,:,:]*100 #m->cm
  
  # https://cloud6.net/so/python/3050836
  data  = np.array(data) #numpy array
  data = np.where(data == 0, np.nan, data)
  data = np.where(data == 9.999000e+20,np.nan,data)
  
  return xx,yy,data
  # print(data.shape)
#---------------
# 2021.08.31 に作成するprogram立ち
local = "/home/ysorimachi/work/hokuriku/dat/snow/111095"
def get_bin_file(ini_j):
  BIN="/home/ysorimachi/work/hokuriku/bin"
  subprocess.run(f"rm {local}/*.bin",shell=True)
  subprocess.run(f"rm {local}/*.nc",shell=True)
  subprocess.run(f"sh {BIN}/get_Snow.sh {ini_j} {local}",shell=True)
  return 
#---------------

def main(ini_j):
  """[summary]
  指定された日時のtelemeの積雪深を解析積雪深から読み取り保存する
  Args:
      ini_j ([type]): [description]
  """
  if 1:
    get_bin_file(ini_j) #data get
  # load snowdepth (numpy 形式)
  try:    
    xx,yy,data = read_111095(ini_j, cate=None,size=None,vmin=0.1,vmax=500)
  # print(xx[0],yy[0], data.shape)

    out_path = f"/work/ysorimachi/hokuriku/dat2/snow_jma/nc/telm_{ini_j}.csv"
    save_depth(xx,yy,data,out_path)
  except:
    print(f"ini_j={ini_j} is Not Found netCDF..")
    log_path = "./jma_111095.log"
    log_write(log_path,f"ini_j={ini_j} is Not Found netCDF..")
  return 


def loop_winter_jtime():
  # _mm = ["201912","202001","202002","202003","202012","202101","202102","202103"]
  # _dd = [31,31,29,31,31,31,28,31]
  _mm = ["202103"]
  _dd = [31]
  _t_all=[]
  for mm,dd in zip(_mm,_dd):
    _t = pd.date_range(start=f"{mm}010000",freq="1H",periods=dd*24)
    _t = [ t.strftime("%Y%m%d%H%M") for t in _t]
    _t_all +=_t
  return _t_all

SNOW_DATASET="/work/ysorimachi/hokuriku/dat2/snow_jma/tmp/telm_snow_depth.csv"
SNOW_DATASET2="/work/ysorimachi/hokuriku/dat2/snow_jma/tmp/telm_snow_rad.csv" #snow with rad
RAD_DATASET="/work/ysorimachi/hokuriku/dat2/snow_jma/tmp/rad_snow.csv"
def reg_DATASET():
  #---------------------------------
  def list_snow_month(df):
    list_mm = sorted(np.unique([t.strftime("%Y%m") for t in df.index]))
    return list_mm
  #---------------------------------
  def premake_dataset(df, tlm):
    """[summary]
    2021.11.17 making tmpfile
    Args:
        df ([type]): [rad and snowdepth]
        tlm ([type]): [teleme Dataframe]

    Returns:
        df[DataFrame]: [rad snowdepth teleme pu]
    """
    if 0: #premake winter rad ----------
      _mm = list_snow_month(df)
      _rad = [ load_rad(month=mm,cate="obs", lag=30, only_mean=False) for mm in _mm]
      rad = pd.concat(_rad,axis=0)
      rad.to_csv(RAD_DATASET)
    rad = pd.read_csv(RAD_DATASET)
    rad["time"] = pd.to_datetime(rad["time"])
    rad = rad.set_index("time")
    _df = []
    
    for code in df.columns:
      obs_point = near_rad(code)
      tmp = pd.concat([rad[obs_point],df[code],tlm[code]],axis=1)
      tmp.columns = ["rad","snow","PV"]
      tmp["pv_max"] = teleme_max(code=code,cate ="max")
      tmp["pv_panel"] = teleme_max(code=code,cate ="panel")
      tmp["snow"] = tmp["snow"].fillna(method="pad")
      tmp["code"] = code
      _df.append(tmp)
      
    df = pd.concat(_df,axis=0)
    return df
  
  def premake_teleme(df):
    if 1: #premake winter rad ----------
      _mm = list_snow_month(df)
      _df = [ load_teleme(month=mm,min=30) for mm in _mm]
      df = pd.concat(_df,axis=0)
    return df
  
  #---------------------------------
  _path = sorted(glob.glob("/work/ysorimachi/hokuriku/dat2/snow_jma/nc/*.csv"))
  print("INIT->",os.path.basename(_path[0]))
  print("END ->",os.path.basename(_path[-1]))
  if 1:
    _df =[]
    _ini_j=[]
    for p in tqdm(_path):
      df = pd.read_csv(p)
      ini_j = os.path.basename(p).split("_")[1][:12]
      _ini_j.append(ini_j)
      df = df.replace(9999,np.nan)[["name","val"]].set_index("name")
      df.name = ini_j
      _df.append(df)
    
    df = pd.concat(_df,axis=1)
    df.columns = _ini_j
    df = df.T 
    df.to_csv(SNOW_DATASET)
  # --- ------------------------- --- # 
  df = pd.read_csv(SNOW_DATASET)
  df["Unnamed: 0"] = pd.to_datetime(df["Unnamed: 0"].astype(str))
  df = df.set_index("Unnamed: 0")
  df.index.name = "time"
  df = df.drop(["telm004","telm040","telm041"],axis=1)

  # --- ------------------------- --- #
  tlm = premake_teleme(df)
  rad = premake_dataset(df,tlm)
  
  rad.to_csv(SNOW_DATASET2)
  return

def reg_snow_PU_Lower():
  df = pd.read_csv(SNOW_DATASET2)
  PNG="/home/ysorimachi/work/hokuriku/dat/rad/reg_snow_pu/png"
  def clensing(df):
    n0 = df.shape[0]
    df = df.dropna()
    df["pu"] = df["PV"] / df["pv_max"]
    df = df[(df["rad"]>600)&(df["rad"]<1200)]
    # df["rad"] /=1000
    n1 = df.shape[0]
    print(f"After Clensing: [{n0}] -> [{n1}]({np.round(n1*100/n0,1)}%) ")
    return df
  
  def set_ax(ax):
    ax.set_ylim(0,1)
    ax.set_xlim(0,ax.get_xlim()[1])
    ax.set_xlim(0,100)
    # ax.set_xlabel(r"OBS-RAD[W/$m^2$]")
    ax.set_xlabel(r"SNOW-DEPTH[cm]")
    ax.set_ylabel(r"PU[-]")
    return ax
  #------------------
  df["pu"] = df["PV"] / df["pv_max"]
  df = clensing(df)
  
  # df = master[master["snow"]>=depth]
  f,ax = plt.subplots(figsize=(10,10))
  cmap,norm = cmap_snow()
  # 3----------------
  # im = ax.scatter(df["rad"],df["pu"], s=df["snow"], c=df["snow"], cmap=cmap,norm=norm,alpha=0.5)
  # im = ax.scatter(df["snow"],df["pu"], s=35, c=df["rad"], cmap="jet",alpha=0.5)
  im = ax.scatter(df["snow"],df["pu"], s=35, color="gray")
  #--------
  ax = set_ax(ax)
  x = np.linspace(ax.get_xlim()[0],ax.get_xlim()[1],1000)
  a,threshold = 0.25,55
  fit = 1/(1 + np.exp(a * (x - threshold )))
  
  print("0 稼働率", snow_PV_rate(x=0))
  print("20 稼働率", snow_PV_rate(x=20))
  print("70 稼働率", snow_PV_rate(x=70))
  ax.plot(x,fit, label="snow-Lower Fit func(sigmoid)", color="r",lw=10, alpha=.5)
  ax.axvline(x=threshold, color="k", lw=1)
  # im = ax.scatter(df["snow"],df["pu"])
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  cbar = plt.colorbar(im, cax=cax)
  cbar.set_label('solar-rad')
  f.savefig(f"{PNG}/pu_depth.png",bbox_inches="tight")
  
  # print("end", depth)
  return

def snow_PV_rate(x,a=0.25,threshold=55):
  """[summary]
  Args:
      x ([type]): [snow depth]
      a (int, optional): [description]. Defaults to 55.
      threshold (float, optional): [description]. Defaults to 0.25.
  Returns:
      [type]: snow PV rate(0~1)
  """
  return 1/(1 + np.exp(a * (x - threshold )))
  
  
  


if __name__ =="__main__":
  # ini_u = sys.argv[1]
  
  # 2021.11.09 冬季期間の積雪深を読み取るデータセットの作成を行う ----
  if 0:
    log_path = "./jma_111095.log"
    log_write(log_path,"[start]",init=True)
    list_jt = sorted(loop_winter_jtime())
    # print(list_jt)
    # sys.exit()
    for i,ini_j in enumerate(list_jt):
      main(ini_j)
      log_write(log_path,f"[end] {ini_j} {i+1}/{len(list_jt)}")
  # main(ini_j)
  
  
# 2021.11.15 冬季期間の積雪深を読み取るデータセッの相関関係を表示する ----
  if 1:
    reg_DATASET() #回帰用のデータセット作成
    # reg_snow_PU_Lower() #回帰係数の算出 
    # tmp_sigmoid()

  
  
  