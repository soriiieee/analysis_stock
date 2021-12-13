# -*- coding: utf-8 -*-
# when   : 2021.09.09
# who : [sori-machi]
# what : 
# *日射量(地上/衛星)、出力、アメダス積雪深の関係性
# *冬季期間における、富山・福井の日射量影響の把握/冬季期間においては、積雪により、日射量が過小評価になりがちなことを、plotlyで確認。
#---------------------------------------------------------------------------
# basic-module
import matplotlib.pyplot as plt
import sys,os,re,glob
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 18
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.simplefilter('ignore')

# <!-- cmap = plt.get_cmap("tab10") # ココがポイント -->
import matplotlib.colors as mcolors
_color = [ mcolors.rgb2hex(plt.cm.get_cmap('tab10')(i)) for i in range(10)]

from tqdm import tqdm
import seaborn as sns
import math
# https://www.python.ambitious-engineer.com/archives/1140
#---------------------------------------------------
# sori -module
sys.path.append('/home/ysorimachi/tool')
from getErrorValues import me,rmse,mae,r2 #(x,y)
# from convSokuhouData import conv_sfc #(df, ave=minutes,hour)
from convAmedasData import conv_amd
#amedas relate 2020.02,04 making...
from tool_AMeDaS import code2name, name2code
from tool_110570 import get_110570,open_110570
from tool_100571 import get_100571,open_100571
#(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
import subprocess
from utils_plotly import plotly_2axis#(df,col1,col2,html_path, title="sampe"):
# outd ='/home/griduser/work/sori-py2/deep/out/0924_01'
# os.makedirs(outd, exist_ok=True)
# subprocess.run('rm -f *.png', cwd=outd, shell=True)
#------------------------
# 2021.09.09 
sys.path.append("..")
from tmp.amedas import get_List
from teleme.reg_PU_001 import load_teleme
from smame.reg_PU_001 import load_smame,get_a_b_c_d #(month,cate) #("surplus","month")
from teleme.utils_teleme import teleme_max#(code=None,cate ="max")
from utils import load_rad#(month="201904",cate="obs", lag=30)

#------------------------
# from mapbox import map_lonlat_multi#(_df,html_path,size=4,zoom=4)
from mapbox import map_lonlat2,map_lonlat# (_df,text="name",html_path,size=4,zoom=4)

DHOME="/work/ysorimachi/hokuriku/snow_hosei/rad210524" #8now0/sfc
DAT_1MIN="/home/ysorimachi/work/hokuriku/dat/rad/111600_2sites"
TMP="/home/ysorimachi/work/hokuriku/tmp/tmp210524"
point_hash = {"cpnt15": "47607","cpnt18": "47616"}
name_hash = {"cpnt15": "TOYAMA","cpnt18": "FUKUI"}

#settig loop
def load_month(isWinter=False):
  if isWinter:
    return list([202012,202101,202102,202103])
  else:
    return list([202004,202005,202006,202007,202008,202009,202010,202011,202012,202101,202102,202103])
  

# set

def check_rad(code,month):
  """
  2021.05.24 
  個別地点、個別の月において、日射量と積雪について確認する
  """
  #local function ---------------
  def clensing(df):
    for c in ["rad","H0","8now0"]:
      df[c] = df[c].apply(lambda x: np.nan if x< 0 or x>1400 else x)
    df = df.dropna(subset=["rad","H0","8now0"])
    return df
  
  def load_sfc(scode,month):
    path = f"{DHOME}/sfc/sfc_10minh_{month}_{scode}.csv"
    df = pd.read_csv(path)
    df = conv_sfc(df, ave=False)
    df["time"] = df["time"].apply(lambda x:x.strftime("%Y%m%d%H%M"))
    df = df[["time","snowDepth"]]
    df = df.replace(9999,np.nan)
    return df
  
  df = pd.read_csv(f"{DHOME}/dataset/{code}_{month}.csv")
  df["time"] = df["time"].astype(str)
  # df = clensing(df)
  scode = point_hash[code]
  ame = load_sfc(scode,month)
  
  # print(df)
  df = df.merge(ame, on="time",how="inner")
  df["time"] = pd.to_datetime(df["time"])
  html_path = f"{TMP}/{code}_{month}_rad_SNOW.html"
  plotly_2axis(df,["rad","8now0"],["snowDepth"],html_path, title=f"{code}_{month}_rad_SNOW")
  print(df.head())
  sys.exit()


def plot_map():
  """
  2021.09.09 : 日射量/アメダス観測点を表示する
  """
  df = get_List()
  df["flg"] = df["name"].isnull()
  df.loc[df["flg"]==False, "cate"] = "日"
  print(df["cate"].unique())
  _df,_text = [],[]
  for i,c in enumerate(df["cate"].unique()):
    tmp = df[df["cate"]==c]
    tmp["color"] = i
    _df.append(tmp)
    _text.append(c)
    
  df = pd.concat(_df,axis=0)
  # df["text"] = df["code"].astype(str) + "-" + df["cate"] + "-" + df["name"]
  df["text"] = df["code"].astype(str) +"-" + df["cate"]
  # print(df.head(50))
  # sys.exit()

  out_d= "/home/ysorimachi/work/hokuriku/out/snow/html"
  html_path = f"{out_d}/map_amedas.html"
  # map_lonlat_multi(_df,_text,html_path)
  map_lonlat(df,html_path =html_path, text="text",size=4,size_col="color",zoom=4)




#---------------------------------
#-----------------
def get_snw(month,code):
  local = "/home/ysorimachi/work/hokuriku/dat/snow/amedas"
  # path = f"{local}/amd_10minh_{month}_{code}.csv"
  # if not os.path.exists(path):
  #   subprocess.run("sh amd_get.sh {} {} {}".format(month,code,local), shell=True)
  # else:
  #   print(f"already get ..{month} {code}")
  if 202004 <= int(month) <= 202006:
    path = f"{local}/snow_2004.csv"
  elif 202007 <= int(month) <= 202009:
    path = f"{local}/snow_2007.csv"
  elif 202010 <= int(month) <= 202012:
    path = f"{local}/snow_2010.csv"
  elif 202101 <= int(month) <= 202103:
    path = f"{local}/snow_2101.csv"
  else:
    path = "not found"
  df = pd.read_csv(path)
  
  for c in df.columns[1:]:
    df[c] = df[c].apply(lambda  x: x if x>0 else 0)
    
  df["time"] = pd.to_datetime(df["time"])
  df = df.set_index("time")
  if code == "55056":
    return df["魚津"]
  if code == "55151":
    return df["富山"]
  if code == "56286":
    return df["白山河内"]

def loop_month(st = "201904", ed="202104"):
  _t = pd.date_range(start = f"{st}300000",end = f"{ed}300000", freq="M")
  _t = [ t.strftime("%Y%m") for t in _t]
  _t = _t[:-1]
  return _t

def get_pv(cate,month,pv_name):
  #--------------
  if cate == "teleme":
    df = load_teleme(month)
    max_val = teleme_max(pv_name)
    
    df["max"] = max_val
    df = df[[pv_name,"max"]]
    df.columns = ["PV","max"]
    return df

  if cate == "surplus":
    # 事前に、/home/ysorimachi/work/hokuriku/py/smame のdetails_smame2.pyを実行して、対象地点のみの合算ファイルを作成しておくことが必要
    DHOME_TMP="/home/ysorimachi/work/hokuriku/dat/snow/csv/tmp_smame_month"
    path = f"{DHOME_TMP}/{cate}_{month}.csv"
    df = pd.read_csv(path)
    df["time"]= df["time"].astype(str)
    df["time"] = df["time"].apply(lambda x: x[0:8] + "0000" if x[8:10] == "24" else x)
    df["time"] = pd.to_datetime(df["time"].astype(str))
    df = df.set_index("time")
    df["sum"] *=2 #30分間隔なので
    # df = df.rename(columns={"sum":"PV"})
    return df[["sum","max"]]


def details_effect(NAME):
  if NAME=="AREA001":
    rad_name,pv_name,snow_code,cate = "unyo001","telm007","55056","teleme"
  if NAME=="AREA002":
    rad_name,pv_name,snow_code,cate = "unyo001","telm007","55056","teleme"
  if NAME=="AREA003":
    rad_name,pv_name,snow_code,cate = "unyo012","telm007","56286","surplus"
    
  return rad_name,pv_name,snow_code,cate

def snow_effect(month="202103"):
  """
  2021 .09.09
  積雪深が個別のPV出力に影響を与えていたのかを調査
  """
  HTML_HOME="/home/ysorimachi/work/hokuriku/out/snow/html"
  # area setting  ---------
  # NAME="AREA001"
  NAME="AREA003"
  radname = "obs"
  
  rad_name,pv_name,snow_code,cate = details_effect(NAME)
  png_title = f"{NAME}({rad_name}/{pv_name}/{snow_code})"
  # print(png_title)
  # sys.exit()
  
  #  mk dataset ---------
  sn_df= get_snw(month,snow_code)# dataget
  pv_df = get_pv(cate,month,pv_name) #teleme&smame

  rad_df = load_rad(month=month,cate=radname, lag=30) #rad_
  df = pd.concat([rad_df[rad_name],pv_df,sn_df],axis=1)
  
  df.columns = ["rad","PV","max","snow"]
  df["snow"] = df["snow"].fillna(method = "pad")
  df = df.dropna()
  
  # calc -----
  df["p.u"] = df["PV"]/df["max"]
  df["rad"] /=1000
  
  DOUT="/home/ysorimachi/work/hokuriku/dat/snow/csv/point"
  df.to_csv(f"{DOUT}/{NAME}_{month}.csv")
  
  if 0: #plotly 
    df = df.reset_index()
    html_path = f"{HTML_HOME}/ts_{NAME}.html"
    plotly_2axis(df,["rad","p.u"],["snow"],html_path, title=png_title, vmax=1)
  
  if 0: #png
    df = df.reset_index()
    png_d ="/home/ysorimachi/work/hokuriku/out/snow/png"
    from plot1m import plot1m_2axis#(df,_col,_sub_col=False,month=False,_ylim=[0,1000,0,100],title=False,step=6)
    f = plot1m_2axis(df,_col=["rad","p.u"],_sub_col=["snow"],month=month,_ylim=[0,1.1,0,120],title=False,step=6)
    f.savefig(f"{png_d}/ts_{month}_{NAME}.png",bbox_inches="tight")
    print(png_d, month)
  return

def plot_pu(NAME):
  """
  2021.09.12
  事前にデータセットを作成して置く必要がある
  """
  DHOME="/home/ysorimachi/work/hokuriku/dat/snow/csv/point"
  png_d ="/home/ysorimachi/work/hokuriku/out/snow/png"
  
  _path = sorted(glob.glob(f"{DHOME}/{NAME}*.csv"))
  _df = [pd.read_csv(path) for path in _path]
  df = pd.concat(_df,axis=0)
  N=df.shape[0]
  
  # f,ax = plt.subplots(1,3,figsize=(18,5))
  f,ax = plt.subplots(figsize=(9,9))
  
  _h = [0,5,20]
  
  ax.scatter(df["rad"],df["p.u"],color="gray", s=1, alpha=0.3,label="全データ")
  for i,h in enumerate(_h):
    tmp = df[df["snow"]>h]
    title = f"積雪別日射量とp.uの関係"
    percent = np.round(tmp.shape[0]*100/N,1)
    color = _color[i]
    if i==2:
      size = 50
      marker="o"
      color="r"
    else:
      size = 12
      marker="o"
    
    ax.scatter(tmp["rad"],tmp["p.u"],color=color, s=size,marker=marker, alpha=1, label=f"SNOW({h}cm超-{percent}[%])")
  
  
  if 1:
    a,b,c,d = get_a_b_c_d("202101","8now0")
    _x = np.linspace(0,1,1000)
    _y = a*_x**3 +b*_x**2 + c*_x + d
    ax.plot(_x, _y, color="g", lw=2, label="p.u回帰曲線(1月)")
    # print(a,b,c,d)
    # sys.exit()
  
  ax.plot(_x, _x, color="k", lw=1)
  ax.set_xlabel("日射量[kW/m2]")  
  ax.set_ylabel("p.u[-]")  
  ax.set_xlim(0,1)  
  ax.set_ylim(0,1)
  ax.set_title(title)
  plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18)
  
  f.savefig(f"{png_d}/scatter_{NAME}.png", bbox_inches="tight")  
  
  print("DIRECTR",png_d)
  sys.exit()
  


if __name__ == "__main__":
  #---------------------------------------
  # #all months -------------
  if 0:
    plot_map()
    
  if 0: #"make dataset 
    for month in loop_month()[12:]:
      # month="202010"
      snow_effect(month=month)
      print(datetime.now(), "[END]", month)
      # sys.exit()
  if 1:
    NAME="AREA001" #teleme
    NAME="AREA003" #smame
    plot_pu(NAME)
  