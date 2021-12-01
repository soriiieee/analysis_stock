# -*- coding: utf-8 -*-
# """
# Created on Tue Oct  2 10:04:51 2018
 
# @author: sakamoto
# """
 
import pandas as pd
import numpy as np
from datetime import datetime
import os,sys
import warnings
warnings.simplefilter("ignore")

rmiss= 9999.

#ubroutine
def conv2all(df):
  for i,col in enumerate(df.columns):
    if i==4: #lat
      df[col] = np.floor(df[col]/1000.) + (df[col]/1000. - np.floor(df[col]/1000.))*100./60
    elif i==5: #lat
      df[col] = np.floor(df[col]/1000.) + (df[col]/1000. - np.floor(df[col]/1000.))*100/60
    elif i==6:
      df[col] = (df[col] - 20000) / 10.
    elif 7 <=i<= 13:
      df[col] = rmiss
    #rain
    elif i==20:
      df[col] =df[col] / 10.
    elif i==22:
      df[col] =df[col] / 10.
    elif i==24:
      df[col] =df[col] / 10.
    elif i==28:
      df[col] =df[col] / 10.
    #wind
    elif i==32:
      df[col] =df[col] / 10.
    elif i==45:
      df[col] =df[col] / 10.
    elif i==47:
      df[col] =df[col] / 10.
    #temp
    elif i==53:
      df[col] =df[col] / 10.
    elif i==55:
      df[col] =df[col] / 10.
    elif i==59:
      df[col] =df[col] / 10.
    
    #soaalar rad
    elif i==66:
      df[col] =df[col] / 10.
    elif i==68:
      df[col] =rmiss
    elif i==70:
      df[col] =rmiss
    elif i==72: #zenten
      df[col] = df[col] / 60.

    #pressure
    elif i==81:
      df[col] = df[col] / 10.
    elif i==83:
      df[col] = df[col] / 10.
    elif i==91:
      df[col] = df[col] / 10.

    elif i==101:
      df[col] = df[col] / 10.
    elif i==103:
      df[col] =rmiss
    elif i==105:
      df[col] =rmiss
    else :
      pass

  return df

def mkTimeColumn(df):
  y0 = df["年"].values[0]
  m0 = df["月"].values[0]
  d0 = df["日"].values[0]
  H0 = df["時"].values[0]
  M0 = df["分"].values[0]

  # print(y0,m0,d0,H0,M0)
  ini_j0 = datetime(y0,m0,d0,H0,M0).strftime("%Y%m%d%H%M")
  _time = pd.date_range(start=ini_j0, freq="10T", periods=len(df))
  return _time

def clensing(df,_col):
  # use_col =['precip', 'windDirection', 'windSpeed',
  #     'temp', 'maxTemp', 'minTemp', 'sunTimeSec', 'sunRadW', 'snowDepth',
  #     'press', 'seaPress', 'humidity']
  for col in _col:
      df[col] = df[col].apply(lambda x: np.nan if np.abs(x) >10000 else x)
      df[col] = df[col].fillna(method = "ffill")
  return df

#-- main 1--- 
def conv2allwithTime(df):
  df = conv2all(df)
  df["time"] = mkTimeColumn(df)
  return df


def mk_average(df,ave):
  if ave==10:
    return df
  elif ave==30:
    lags=3
  elif ave ==60:
    lags=6
  elif ave==24:
    lags=6*26
  else:
    lags=0
  
  # add
  for col in ['tenminPrecip','tenminSunshineTime']:
    df[col] = df[col].rolling(lags).sum()
  
  #average
  for col in ['windSpeed','temp', 'tenminMaxTemp', 'tenminMinTemp','tenminSunshine','sixtyminSunshine','stationPressure', 'seaLevelPressure','humidity']:
    df[col] = df[col].rolling(lags).mean().apply(lambda x: np.round(x, 2))
  
  df = df.iloc[df.index % lags==2,:]
  df = df.reset_index(drop=True)
  return df



#-- main 2--- 
def conv2CutCols(df, ave=None, snow=False):
  # df = conv2allwithTime(df)
  use_col=['time','緯度', '経度', '標高','10分間降水量','1時間降水量','平均風向(前10分間のベクトル平均)16方位', '平均風速(10分移動平均)', '気温', '最高気温(前10分間)', '最低気温(前10分間)','10分間日照時間','全天日射量', '1時間全天日射量','積雪の深さ', '現地気圧', '海面気圧','相対湿度']
  df= df[use_col]
  
  use2_col = ['time','lat', 'log', 'z','tenminPrecip','sixtyminPrecip','windDirection', 'windSpeed','temp', 'tenminMaxTemp', 'tenminMinTemp','tenminSunshineTime','tenminSunshine', 'sixtyminSunshine','snowDepth', 'stationPressure', 'seaLevelPressure','humidity']

  df.columns = use2_col

  if ave:
    df = mk_average(df,ave)
    if snow:
      use_col_snow = ['time','precip','temp','humidity','snowDepth','sunTimeSec','windSpeed']
      return df[use_col_snow]
    return df
  return df


def conv_sfc(df, ave=30):
  df = conv2allwithTime(df)
  df = conv2CutCols(df, ave=ave, snow=False)
  return df



if __name__=="__main__":
  path =f"/work/ysorimachi/make_SOKUHOU3/out/202102/sfc2/sfc_10minh_202102_47622.csv"
  df = pd.read_csv(path)
  
  df = conv_sfc(df)
  print(df.head())
  sys.exit()
  with open("/home/ysorimachi/work/make_SOKUHOU3/tmp/sfc_colums.dat", "w") as f:
    for i ,col in enumerate(df.columns):
      text = f"{str(i).zfill(4)} ,{col}\n"
      f.write(text)
  sys.exit()
  
