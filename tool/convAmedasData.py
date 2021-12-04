# -*- coding: utf-8 -*-
# """
# Created on Tue Oct  2 10:04:51 2018
 
# @author: sakamoto
# """
 
import pandas as pd
import numpy as np
from datetime import datetime
import sys
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
    elif 7 <=i<= 10:
      df[col] = rmiss
    #rain
    elif i==17:
      df[col] =df[col] / 10.
    elif i==19:
      df[col] =df[col] / 10.
    elif i==21:
      df[col] =df[col] / 10.
    elif i==25:
      df[col] =df[col] / 10.
    #wind
    elif i==29:
      df[col] =df[col] / 10.
    elif i==42:
      df[col] =df[col] / 10.
    elif i==44:
      df[col] =df[col] / 10.
    #temp
    elif i==50:
      df[col] =df[col] / 10.
    elif i==52:
      df[col] =df[col] / 10.
    elif i==56:
      df[col] =df[col] / 10.
    
    #soaalar rad
    elif i==61: #日照時間
      df[col] = df[col]

    #snow
    elif i==65:
      df[col] = df[col]
    else:
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


#-- main 1--- 
def conv2allwithTime(df):
  df = conv2all(df)
  df["time"] = mkTimeColumn(df)
  return df


def mk_average(df,ave):
  if ave==30:
    lags=3
  elif ave ==60:
    lags=6
  elif ave==24:
    lags=6*26
  else:
    lags=0
  
  # add
  for col in ['tenminPrecip','tenminSunshineTime']:
    df[col] = df[col].rolling(lags).sum().fillna(0)
    # # df[col] = df[col].rolling(lags).sum()
    # print(df[col].isnull().sum())
    # print(col)
    # print(df[col].max())
    # print(df[col].min())
    # sys.exit()
  
  #average
  for col in ['windSpeed','temp', 'tenminMaxTemp', 'tenminMinTemp','snowDepth']:
    df[col] = df[col].rolling(lags).mean().apply(lambda x: np.round(x, 2)).fillna(0)
  
  df = df.iloc[df.index % lags==2,:]
  df = df.reset_index(drop=True)
  return df



#-- main 2--- 
def conv2CutCols(df, ave=None):
  # df = conv2allwithTime(df)
  use_col=['time','緯度', '経度', '標高','10分間降水量','1時間降水量','平均風向(前10分間のベクトル平均)16方位', '平均風速(10分移動平均)', '気温', '最高気温(前10分間)', '最低気温(前10分間)','10分間日照時間','積雪の深さ']
  df= df[use_col]


  use2_col = ['time','lat', 'lon', 'z','tenminPrecip','sixtyminPrecip','windDirection', 'windSpeed','temp', 'tenminMaxTemp', 'tenminMinTemp','tenminSunshineTime','snowDepth']
  df.columns = use2_col

  # print(len(use_col))
  # print(len(use2_col))
  # sys.exit()

  if ave:
    df = mk_average(df,ave)
    # if ave==30:
    # use2_col = ['time','lat', 'lon', 'z','Precip','windDirection', 'windSpeed','temp', 'MaxTemp', 'MinTemp','SunshineTimeSeconds','snowDepth']
    # df.columns = use2_col
    return df

  return df


def conv_amd(df, ave=30):
  df = conv2allwithTime(df)
  df = conv2CutCols(df, ave=ave)
  return df



if __name__ =="__main__":
  DIR="/work/griduser/tmp/ysorimachi/snowdepth_calc200525/dat0701/201401"
  tmp = pd.read_csv(f"{DIR}/amd_10minh_201401_11016.csv")
  # print(tmp.head())
  # for i ,col in enumerate(tmp.columns):
  #   print(i,col)
  df = conv2allwithTime(tmp)
  df = conv2CutCols(df, ave=30)
  print(df.head())
  sys.exit()

