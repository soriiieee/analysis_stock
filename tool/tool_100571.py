# -*- coding: utf-8 -*-
# xml形式のデータから日射量を切り出す
"""
params : 
書き込みファイルのあるディレクトリ
取得する時間の日本時間

"""
import xml.etree.ElementTree as ET
import numpy as np
import sys
import os
import requests
from tqdm import tqdm
import pandas as pd
import time

from datetime import datetime, timedelta

from tool_AMeDaS import code2name, name2code

xml_col1=["stationLongitude","stationLatitude","stationHeightASL","anemometerHeight"]
xml_col2=["tenminPrecip","sixtyminPrecip","windDirection","windSpeed","temp","tenminSunshine","sixtyminSunshine", "snowDepth","humidity","seaLevelPressure", "stationPressure","visibility","dailyMaxTenminPrecip", "dailyMaxSixtyminPrecip","dailyMaxWindDirection","dailyMaxInstWindSpeed","dailyMaxInstWindDirection", "dailyMaxTemp","dailyMaxSnowDepth","dailyMinTemp", "dailyMinHumidity", "onehourMaxTenminPrecip","onehourMaxSixtyminPrecip","onehourMaxWindSpeed","onehourMaxWindDirection", "onehourMaxInstWindSpeed","onehourMaxInstWindDirection","onehourMaxTemp", "onehourMaxSnowDepth","onehourMinTemp","twentyfourhourPrecip","twentyfourhourSnowFall","onehourSnowFall","autoObsWeather","tenminMaxTemp", "tenminMaxInstWindSpeed","tenminMaxInstWindDirection","tenminMinTemp","tenminMinHumidity"]


def make_dummy():
  _inf = [ 9999 for i in range(len(xml_col1))]
  _val = [ 9999 for i in range(len(xml_col2))]
  return _inf + _val

def read_val(e):
  #init
  _inf, _val= [],[] #xml_col1
  a = e.find('observation')

  for epl in xml_col1:
    val = a.find(epl).text
    if val is None:
      val = 9999
    _inf.append(val)

  #weather info2
  for ele in xml_col2:
    val = a.find(ele).text
    if val is None:
      val = 9999
    # val
    _val.append(val)
    
  return _inf + _val


def read_val2(e, val):
  a = e.find('observation')
  v = a.find(val).text

  if v is None:
    v = 9999
  return v

def write(code,ini_j,out_d,data_list):
    # info = [e_codes,e_info1,err_name,err_info,time_name,time_info]
    #info
  L = [str(ele) for ele in data_list ]
  L =" ".join(L)
  f=open(out_d+"/"+code+"_100571.dat","+a") # param[2] = 出力ファイル
  string = "{0} {1}\n".format(ini_j,L)
  f.write(string)
  f.close()
  return



def get_100571(code,_ini_j,out_d,run_mode="RETRY"):
  """
  2021.07.03
  update list datatime...
  """
  for ini_j in _ini_j:
    ini_u = (pd.to_datetime(str(ini_j)) - timedelta(hours=9)).strftime("%Y%m%d%H%M")
    yy,mm,dd,hh,mi = ini_u[0:4],ini_u[4:6],ini_u[6:8],ini_u[8:10],ini_u[10:12]
    yy_j,mm_j,dd_j = ini_j[:4], ini_j[4:6], ini_j[6:8]
    T = 0 if int(ini_u[10:12])==0 else 1
    ff=code[:2]
    
    if run_mode == "RETRY":
      url = f"http://micproxy2.core.micos.jp/stock/{yy}/{mm}/{yy}{mm}{dd}/data/100571/{yy}{mm}{dd}/000{T}{ff}/0000/100571-000{T}{ff}-0000-{ini_u}00.xml"
    # print(url)
      res = requests.get(url,auth=('micosguest', 'mic6guest'))
    if run_mode == "UNYO":
      url = f"http://micproxy1.core.micos.jp/product/data/100571/{yy}{mm}{dd}/000{T}{ff}/0000/100571-000{T}{ff}-0000-{ini_u}00.xml"
    # print(url)
      res = requests.get(url)
      
    isGEt=0
    if res.status_code == 200: #sokuhou
      root = ET.fromstring(res.text)
      for e in root.findall('point'):
        if e.attrib["pointCode"]==code:
          data_list = read_val(e)
          isGEt=1
        else:
          pass
    
    
    if isGEt==0:
      data_list = make_dummy()
    write(code,ini_j,out_d,data_list)
  return

def get_100571_val(code,ini_j,val="snowDepth"):
  ini_u = (pd.to_datetime(str(ini_j)) - timedelta(hours=9)).strftime("%Y%m%d%H%M")
  yy,mm,dd,hh,mi = ini_u[0:4],ini_u[4:6],ini_u[6:8],ini_u[8:10],ini_u[10:12]
  yy_j,mm_j,dd_j = ini_j[:4], ini_j[4:6], ini_j[6:8]
  T = 0 if int(ini_u[10:12])==0 else 1
  ff=code[:2]
  
  url = f"http://micproxy2.core.micos.jp/stock/{yy}/{mm}/{yy}{mm}{dd}/data/100571/{yy}{mm}{dd}/000{T}{ff}/0000/100571-000{T}{ff}-0000-{ini_u}00.xml"
  # print(url)
  # sys.exit()
  res = requests.get(url,auth=('micosguest', 'mic6guest'))
  isGEt=0
  if res.status_code == 200: #sokuhou
    root = ET.fromstring(res.text)
    for e in root.findall('point'):
      if e.attrib["pointCode"]==code:
        v = read_val2(e,val)
        isGEt=1
        return v
      else:
        pass
  return 9999

def open_100571(code,path):
  names=["time"]+xml_col1+xml_col2
  df=pd.read_csv(path,delim_whitespace=True,header=None, names=names)
  # df.to_csv(out_csv, index=False)
  return df


if __name__=="__main__":
  out_d="/home/ysorimachi/data/hokuriku/tmp"
  code = "55022"
  print(code2name(code))

  # get_100571("44132","202002090000",out_d)

  path = f"/home/ysorimachi/work/hokuriku/out/smame/ame/{code}_100571.dat"
  csv_path=f"/home/ysorimachi/work/hokuriku/out/smame/ame_csv/{code}_100571.csv"
  open_100571(code,path,csv_path)