# -*- coding: utf-8 -*-
# when   : 2020.0x.xx
# who : [sori-machi]
# what : [ ]
#---------------------------------------------------------------------------
# basic-module
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
_color = [ mcolors.rgb2hex(plt.cm.get_cmap('tab10')(i)) for i in range(10)]

import sys,os,re,glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
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
# os.makedirs(outd, exist_ok=True)
# subprocess.run('rm -f *.png', cwd=outd, shell=True)

MY_ENV="/home/ysorimachi/work/sori_py2"

HOME=f"{MY_ENV}/COVID-19"
SOME=f"{MY_ENV}/analysis_stock"
OUT=f"{SOME}/out"
os.makedirs(OUT, exist_ok=True)

DAYLY_DATA=f"{HOME}/csse_covid_19_data/csse_covid_19_daily_reports"
TS_DATA=f"{HOME}/csse_covid_19_data/csse_covid_19_time_series"

from urllib.request import urlopen
import json
import pandas as pd
import plotly
import plotly.express as px


#-----------------------------------------------------------------------------
def check_dd(dd):
  # yy,mm,dd = list(map(int, [dd[:4], dd[4:6], dd[6:]]))
  yy,mm,dd = dd[:4], dd[4:6], dd[6:]
  path = f"{DAYLY_DATA}/{mm}-{dd}-{yy}.csv"
  df = pd.read_csv(path)
  all_cols = ['FIPS', 'Admin2', 'Province_State', 'Country_Region', 'Last_Update',
       'Lat', 'Long_', 'Confirmed', 'Deaths', 'Recovered', 'Active',
       'Combined_Key', 'Incident_Rate', 'Case_Fatality_Ratio' ]
  use_cols = ['FIPS','Province_State','Country_Region', 'Confirmed', 'Deaths', 'Recovered','Incident_Rate', 'Case_Fatality_Ratio',"Active"]
  
  df  =df[use_cols]
  # df = df.sort_values("Deaths", ascending=False)
  # df = df.sort_values("Incident_Rate", ascending=False)
  # df = df.sort_values("Case_Fatality_Ratio", ascending=False)
  
  df = df.groupby("Country_Region").agg({
    "Confirmed": "sum",
    "Deaths": "sum",
    "Recovered": "sum",
    })
  df = df.sort_values("Deaths", ascending=False)
  df = df.reset_index()
  return df

def load_tbl(ONLY_NATION=True):
  path = f"{HOME}/csse_covid_19_data/UID_ISO_FIPS_LookUp_Table.csv"
  df = pd.read_csv(path)
  use_col = ["UID","iso3","Combined_Key","Population"]
  df = df[use_col]
  if ONLY_NATION:
    df = df[df["UID"]<1000] #地域等々を除外して国のみ抽出する
  # print(df[df["iso3"]=="JPN"])
  # print(df[df["iso3"]=="USA"])
  return df

def check_dd2(dd):
  """ 緯度経度情報も併せて表示するようなprogram """
  # yy,mm,dd = list(map(int, [dd[:4], dd[4:6], dd[6:]]))
  yy,mm,dd = dd[:4], dd[4:6], dd[6:]
  path = f"{DAYLY_DATA}/{mm}-{dd}-{yy}.csv"
  df = pd.read_csv(path)
  df = df.merge(load_tbl(ONLY_NATION=False), on="Combined_Key",how="left")
  # print(df.head())
  # sys.exit()
  all_cols = ['FIPS', 'Admin2', 'Province_State', 'Country_Region', 'Last_Update',
       'Lat', 'Long_', 'Confirmed', 'Deaths', 'Recovered', 'Active',
       'Combined_Key', 'Incident_Rate', 'Case_Fatality_Ratio' ]
  use_cols = ['FIPS','Province_State','Country_Region', 'Confirmed', 'Deaths', 'Recovered','Lat', 'Long_',"iso3","Population","Combined_Key"]
  
  df  =df[use_cols]
  # df = df.sort_values("Deaths", ascending=False)
  # df = df.sort_values("Incident_Rate", ascending=False)
  # df = df.sort_values("Case_Fatality_Ratio", ascending=False)
  #--------
  # df = df.groupby("iso3").agg({
  #   "Confirmed": "sum",
  #   "Deaths": "sum",
  #   "Recovered": "sum",
  #   "Lat": "mean",
  #   "Long_": "mean",
  #   })
  # df = df.sort_values("Deaths", ascending=False)
  # df = df.reset_index()
  #--------
  # out_col = ['Confirmed', 'Deaths', 'Recovered',"iso3"]
  out_col = ['Confirmed', 'Deaths', 'Recovered',"iso3","Lat","Long_"]
  df = df.sort_values("Deaths", ascending=False)
  df = df[out_col]
  return df


def change_ratio(df, ratio="pct"):
  drop_col = ["c1","d1","r1","c0","d0","r0"]
  if ratio == "pct":
    df["Confirmed"] = (df["c1"] - df["c0"])/df["c0"]
    df["Deaths"] = (df["d1"] - df["d0"])/df["d0"]
    df["Recovered"] = df["r1"] - df["r0"]
  if ratio == "diff":
    df["Confirmed"] = df["c1"] - df["c0"]
    df["Deaths"] = df["d1"] - df["d0"]
    df["Recovered"] = df["r1"] - df["r0"]
  df = df.drop(drop_col, axis=1)
  return df


def select_dd(dd="20211025", N=30, cate="Confirmed",ratio="diff", return_df=False):
  """
  2021.11.03 
  """
  # 1週間前の　情報と照らし合わせて増加率の多い国を抽出するような処理
  dd2 = dtinc_dd(dd,-7)
  # print(dd2,"-", dd)
  # df = check_dd(dd=dd).set_index("Country_Region")
  # df2 = check_dd(dd=dd2).set_index("Country_Region")
  df = check_dd2(dd=dd).set_index(["iso3","Lat","Long_"])
  df2 = check_dd2(dd=dd2).set_index(["iso3","Lat","Long_"])
  # df = check_dd2(dd=dd).set_index("iso3")
  # df2 = check_dd2(dd=dd2).set_index("iso3")
  
  df = pd.concat([df,df2],axis=1)
  print(df.head())
  # sys.exit()
  drop_col = ["c1","d1","r1","c0","d0","r0"]
  df.columns = drop_col
  
  df = change_ratio(df, ratio=ratio)
  df = df.sort_values(cate, ascending=False)
  # df = df.drop(drop_col, axis=1)
  
  if return_df:
    return df
  
  df.index.name = "Country_Region"
  df = df.reset_index()
  _nation = df["Country_Region"].values[:N].tolist()
  return _nation

def select_dd2(dd="20211025",freq=7, n=5, list_ele=['Confirmed', 'Deaths', 'Recovered']):
  """
  2021.11.03 
  2021.12.02 100万人あたりの増加数
  """
  # print(dd2,"-", dd)
  # df = check_dd(dd=dd).set_index("Country_Region")
  # df2 = check_dd(dd=dd2).set_index("Country_Region")
  # df = check_dd2(dd=dd).set_index(["iso3","Lat","Long_"]) #2021.11.03
  # ------------------------
  _ndf=[]
  for i in range(n):
    df = check_dd2(dd= dtinc_dd(dd,-i*freq))
    df = df.groupby("iso3").sum() # today
    _ndf.append(df)
  #-------------------------
  _df_c = []
  for c in list_ele:
    _df=[]
    for i,df in enumerate(_ndf):
      s = df[c]
      lbl = dtinc_dd(dd,-i*freq)[4:8]
      s.name = f"{c}-{i}(~{lbl})"
      _df.append(s)
    df_c = pd.concat(_df,axis=1)
    tbl = load_tbl().set_index("iso3")
    df_c = pd.concat([df_c,tbl],axis=1)
    df_c = df_c.sort_values("Population",ascending=False)
    
    #----------------- 人口補正　---------
    use_col = [ c1 for c1 in df_c.columns if c in c1 ]
    for c2 in use_col:
      df_c[c2] = df_c[c2] / df_c["Population"] * 1e6 #人口100万人当たりの数
    #----------------- 人口補正　---------
    
    _df_c.append(df_c)
    
  return _df_c
  
#-----------------------------------------------------------------------------
def conv_time(x):
  mm,dd,yy = x.split("/")
  yy = "20"+yy
  yy = int(yy)
  mm = int(mm)
  dd = int(dd)
  t = datetime(yy,mm,dd,0,0)
  return t


def cut_time(df,st,ed):
  if ed:
    df = df[st:ed]
  else:
    df = df[st:]
  return df

def dtinc_dd(dd,delta):
  yy,mm,day  = map(int, [dd[:4],dd[4:6],dd[6:8]])
  dd2 = (datetime(yy,mm,day,0,0) + timedelta(days=delta)).strftime("%Y%m%d")
  return dd2


def main(dd="20211130"):
  # with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
  #   counties = json.load(response)
    
  df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/fips-unemp-16.csv",
                   dtype={"fips": str})
  
  #---Now ---
  df= select_dd(dd = dd, N=30, cate="Confirmed", ratio="diff",return_df=True)
  print(df.head())
  sys.exit()
  
  # df = select_dd(dd = dtinc_dd(dd,-7), N=30, cate="Confirmed", return_df=True)
  tbl = load_tbl() 
  df1 = df.merge(tbl, on="iso3", how="left").dropna()

  df= select_dd(dd =dtinc_dd(dd,-14), N=30, cate="Confirmed", ratio="diff",return_df=True).reset_index()
  # df = select_dd(dd = dtinc_dd(dd,-7), N=30, cate="Confirmed", return_df=True)
  tbl = load_tbl() 
  df2 = df.merge(tbl, on="iso3", how="left").dropna()
  
  df1 = df1.set_index(["iso3","Lat","Long_","Combined_Key","Population"])
  df2 = df2.set_index(["iso3","Lat","Long_","Combined_Key","Population"])
  df = pd.concat([df1,df2],axis=1)
  drop_col = ["c1","d1","r1","c0","d0","r0"]
  df.columns = drop_col
  # df = change_ratio(df, ratio="pct")
  df = df.reset_index()
  df = df[df["iso3"]=="JPN"]
  print(df.head())
  sys.exit()
  # df = df.reset_index()
  
  col = "Confirmed"
  vmin,vmax = df.describe()[col]["min"],df.describe()[col]["max"]
  vmin,vmax = -0.5,0.5
  
  
  print(df[df["iso3"]=="JPN"])
  print(df.shape)
  sys.exit()
  # print(df.describe())
  # # print(df.head())
  # sys.exit()
  
  # print(df.sort_values(col,ascending=False))
  # sys.exit()
  # print(vmin,vmax)
  # sys.exit()
  # print(df.shape)
  # print(df.head())
  # print(dir(plotly.express.colors.sequential))
  # ['Aggrnyl', 'Aggrnyl_r', 'Agsunset', 'Agsunset_r', 'Blackbody', 'Blackbody_r', 'Bluered', 'Bluered_r', 'Blues', 'Blues_r', 'Blugrn', 'Blugrn_r', 'Bluyl', 'Bluyl_r', 'Brwnyl', 'Brwnyl_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'Burg', 'Burg_r', 'Burgyl', 'Burgyl_r', 'Cividis', 'Cividis_r', 'Darkmint', 'Darkmint_r', 'Electric', 'Electric_r', 'Emrld', 'Emrld_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'Hot', 'Hot_r', 'Inferno', 'Inferno_r', 'Jet', 'Jet_r', 'Magenta', 'Magenta_r', 'Magma', 'Magma_r', 'Mint', 'Mint_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'Oryel', 'Oryel_r', 'Peach', 'Peach_r', 'Pinkyl', 'Pinkyl_r', 'Plasma', 'Plasma_r', 'Plotly3', 'Plotly3_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuRd', 'PuRd_r', 'Purp', 'Purp_r', 'Purples', 'Purples_r', 'Purpor', 'Purpor_r', 'Rainbow', 'Rainbow_r', 'RdBu', 'RdBu_r', 'RdPu', 'RdPu_r', 'Redor', 'Redor_r', 'Reds', 'Reds_r', 'Sunset', 'Sunset_r', 'Sunsetdark', 'Sunsetdark_r', 'Teal', 'Teal_r', 'Tealgrn', 'Tealgrn_r', 'Viridis', 'Viridis_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', '__all__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', '_cols', '_contents', '_k', '_swatches', 'algae', 'algae_r', 'amp', 'amp_r', 'deep', 'deep_r', 'dense', 'dense_r', 'gray', 'gray_r', 'haline', 'haline_r', 'ice', 'ice_r', 'matter', 'matter_r', 'solar', 'solar_r', 'speed', 'speed_r', 'swatches', 'tempo', 'tempo_r', 'thermal', 'thermal_r', 'turbid', 'turbid_r']
  # sys.exit()

  # sys.exit()
  #------------------
  fig = px.choropleth(df,locations = "iso3",
                          color=col,#Deaths
                          hover_name="Confirmed",
                          range_color=[vmin,vmax],
                          # animation_frame='date',
                          projection="natural earth",
                          color_continuous_scale='RdBu_r'
                          # https://plotly.com/python/builtin-colorscales/
                          )
  # sys.exit()
  # fig.show()
  # fig.write_image("../out/nation_map/sample.png")
  DIR=f"{OUT}/covid/nation_map"
  os.makedirs(DIR, exist_ok=True)
  fig.write_html(f"{DIR}/covid_map.html")
  return
  
  
if __name__ == "__main__":
  main()