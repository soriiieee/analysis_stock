# -*- coding: utf-8 -*-
# when   : 2020.0x.xx
# who : sori-machi
# what : 
#---------------------------------------------------------------------------
# basic-module
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
import numpy as np
import itertools
import math
#---------------------------------------------------
# sori -module
sys.path.append('/home/griduser/tool')
import tool_err  # def mk_err(x,y,cutlow,cuthigh):
# import tool_label # def label0(string,num)
# import tool_contour as contour
# import tool_area_dfs
from tqdm import tqdm
# from postToSlack import postToSlack
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
#----------------------------

# postToSlack("START: {}".format(FILE))
# command
# python m22_00.py "/work/griduser/tmp/ysorimachi/obs/ftp" "/work/griduser/tmp/ysorimachi/obs/ftp_out" 201301

usecols = ['緯度', '経度', '標高', '年', '月', '日', '時', '分','10分間降水量', '1時間降水量', '最大10分間降水量(前10分間)', '最大1時間降水量(前10分間)', '最大瞬間風速(前10分間)','平均風速(10分移動平均)', '最大風速(前10分間)','気温','最高気温(前10分間)', '最低気温(前10分間)','10分間日照時間','全天日射量','積雪の深さ','現地気圧', '相対湿度']

usecols2 = ['lat', 'lon', "z", "yy", 'mm', 'dd', 'hh', 'mi','pr_10min', 'pr_1h', 'pr_10min_max', 'pr_1h_max', 'wind_inst','wind', 'wind_max','temp','temp_max','temp_min','suntime','sunrad','snwdepth','hPa', 'humi']

# usecols2 = ['lat', 'lon', "z", "yy", 'mm', 'dd', 'hh', 'mi','pr_10min', 'pr_1h', 'pr_10min_max', 'pr_1h_max', 'wind_inst','wind', 'wind_max','temp','temp_max', 'temp_min','sunrad','snwdepth','hPa', 'humi']

class MakeSokuhouDf:

  def __init__(self,path,yy,mm):
    self.path = path
    self.yy = yy
    self.mm = mm

    ini_j= self.yy + self.mm + "010010" #201811010010
    df = pd.read_csv(self.path)
    df = df[usecols]
    df.columns = usecols2
    _time = pd.date_range(start=ini_j, freq="10T", periods=len(df))
    df["time0"] = _time
    self.df = df
    # return 
    #   
  def cleanData(self):
  # df["lat"] = np.floor(df["lat"]/1000) + (df["lat"] / 1000- np.floor(df["lat"]/1000))*60
    self.df["lat"] =np.floor(self.df["lat"]/1000)+ (self.df["lat"] / 1000- np.floor(self.df["lat"]/1000)) *100 / 60
    self.df["lon"] =np.floor(self.df["lon"]/1000)+ (self.df["lon"] / 1000- np.floor(self.df["lon"]/1000)) *100 / 60
    self.df["z"] = (self.df["z"] -20000)/10
    #pr
    self.df["pr_10min"] = self.df["pr_10min"] /10.
    self.df["pr_1h"] = self.df["pr_1h"] /10.
    self.df["pr_10min_max"] = self.df["pr_10min_max"] /10.
    self.df["pr_1h_max"] = self.df["pr_1h_max"] /10.
    #wind
    self.df["wind_inst"] = self.df["wind_inst"] /10.
    self.df["wind"] = self.df["wind"] /10.
    self.df["wind_max"] = self.df["wind_max"] /10.
    #temp
    self.df["temp"] = self.df["temp"] /10.
    self.df["temp_max"] = self.df["temp_max"] /10.
    self.df["temp_min"] = self.df["temp_min"] /10.
    #sunrad
    self.df["sunrad"] = self.df["sunrad"] / 60.
    self.df["hPa"] = self.df["hPa"] /10.
    self.df["humi"] = self.df["humi"] / 100
    
    return self

  def aveSlice(self,ave):
      #pr
    self.df["pr_10min"] = self.df["pr_10min"].rolling(ave).sum()
    self.df["pr_1h"] = self.df["pr_1h"].rolling(ave).sum()
    self.df["pr_10min_max"] = self.df["pr_10min_max"].rolling(ave).sum()
    self.df["pr_1h_max"] = self.df["pr_1h_max"].rolling(ave).sum()
    #wind
    self.df["wind_inst"] = self.df["wind_inst"].rolling(ave).mean()
    self.df["wind"] = self.df["wind"].rolling(ave).mean()
    self.df["wind_max"] = self.df["wind_max"].rolling(ave).mean()
    #temp
    self.df["temp"] = self.df["temp"].rolling(ave).mean()
    self.df["temp_max"] = self.df["temp_max"].rolling(ave).mean()
    self.df["temp_min"] = self.df["temp_min"].rolling(ave).mean()
    #sunrad
    self.df["sunrad"] = self.df["sunrad"].rolling(ave).mean()
    self.df["suntime"] = self.df["suntime"].rolling(ave).sum()
    self.df["hPa"] = self.df["hPa"].rolling(ave).mean()
    self.df["humi"] = self.df["humi"].rolling(ave).mean()

    self.df =  self.df.iloc[self.df.index % ave== ave-1,:]
    self.df = self.df.reset_index(drop=True)
    return self.df


if __name__ =="__main__":
  code="47662"
  yy = "2018"
  mm="11"

  path = "/home/griduser/work/sori-py2/kaggle/input/price_power/sfc/{0}/sfc_10minh_{1}{2}_{0}.csv".format(code,mm,yy)

  df = mkFile(path, yy,mm)
  print(df.head())

  sys.exit()


  # for code in tqdm(_code):
  #   if fileCheck(code,_month2[yy]):
  #     _df=[]
  #     for month in tqdm(_month2[yy]):
  #       logger.info("MONTH START ! {}".format(month))
  #       DIR = "/work/griduser/tmp/ysorimachi/obs/ftp/{}".format(month)
        
  #       df = pd.read_csv("{}/sfc_10minh_{}_{}.csv".format(DIR,month,code))
  #       df = df[usecols]
  #       df.columns=usecols2
  #       #making data 
  #       df0 = make_data(df)

  #       _df.append(df0)
      
  #     df_all = pd.concat(_df,axis=0).reset_index(drop=True)
  #     logger.info("-----  {} code calc start ! {}------- ".format(month, code))


  #     #making average
  #     df1 = df_all
  #     df2= make_ave(df_all,3)
  #     df3 = make_ave(df_all,6)
  #     df4= make_ave(df_all,24*6)