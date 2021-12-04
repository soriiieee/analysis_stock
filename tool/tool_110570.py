# -*- coding: utf-8 -*-
# xml形式のデータから日射量を切り出す
import xml.etree.ElementTree as ET
import numpy as np
import sys
import os
import requests
from tqdm import tqdm
from datetime import date, datetime, timedelta
import pandas as pd
import time


import subprocess
from tool_AMeDaS import code2name, name2code
# initial setting-----------------------------------

def make_dummy():
  _val = [ 9999 for i in range(4)]
  return _val

def read_val(e):
  _val =[] #xml_col1
  # print(code)
  a = e.find('observation')
  for ele in ["sixtyminGlobalRadiationJ","sixtyminGlobalRadiationW","tenminGlobalRadiationJ","tenminGlobalRadiationW"]:
    val = a.find(ele).text
    if val is None:
      val = 9999
    _val.append(val)
  return _val

def write(code,ini_j,out_d,data_list):
  L = [str(ele) for ele in data_list ]
  L =" ".join(L)
  with open(out_d+"/"+code+"_110570.dat","+a") as f:
    string = "{0} {1}\n".format(ini_j,L)
    f.write(string)
  return


def get_110570(code,ini_j,out_d):
  ini_u = (pd.to_datetime(str(ini_j)) - timedelta(hours=9)).strftime("%Y%m%d%H%M")
  yy,mm,dd,hh,mi = ini_u[0:4],ini_u[4:6],ini_u[6:8],ini_u[8:10],ini_u[10:12]
  yy_j,mm_j,dd_j = ini_j[:4], ini_j[4:6], ini_j[6:8]
  T = 0 if int(ini_u[10:12])==0 else 1
  ff=code[:2]

  url = f"http://micproxy2.core.micos.jp/stock/{yy}/{mm}/{yy}{mm}{dd}/data/110570/{yy}{mm}{dd}/000{T}{ff}/0000/110570-000{T}{ff}-0000-{ini_u}00.xml"
  # time.sleep(0.0)
  res = requests.get(url,auth=('micosguest', 'mic6guest'))
  isGEt=0
  if res.status_code == 200:
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

def open_110570(code,path):
  names=["time"]+ ["sixtyminGlobalRadiationJ","sixtyminGlobalRadiationW","tenminGlobalRadiationJ","tenminGlobalRadiationW"]
  df=pd.read_csv(path,delim_whitespace=True,header=None, names=names)
  # df.to_csv(out_csv, index=False)
  return df


def get_110570(code,ini_j):
  ini_u = (pd.to_datetime(str(ini_j)) - timedelta(hours=9)).strftime("%Y%m%d%H%M")
  yy,mm,dd,hh,mi = ini_u[0:4],ini_u[4:6],ini_u[6:8],ini_u[8:10],ini_u[10:12]
  yy_j,mm_j,dd_j = ini_j[:4], ini_j[4:6], ini_j[6:8]
  T = 0 if int(ini_u[10:12])==0 else 1
  ff=code[:2]

  url = f"http://micproxy2.core.micos.jp/stock/{yy}/{mm}/{yy}{mm}{dd}/data/110570/{yy}{mm}{dd}/000{T}{ff}/0000/110570-000{T}{ff}-0000-{ini_u}00.xml"
  # time.sleep(0.0)
  res = requests.get(url,auth=('micosguest', 'mic6guest'))
  isGEt=0
  if res.status_code == 200:
    root = ET.fromstring(res.text)
    for e in root.findall('point'):
      if e.attrib["pointCode"]==code:
        data_list = read_val(e)
        isGEt=1
      else:
        pass
  
  if isGEt==0:
    data_list = make_dummy()
  # write(code,ini_j,out_d,data_list)
  return data_list


if __name__ == "__main__":
  out_d="/home/ysorimachi/data/hokuriku/tmp"
  # get_110570("44132","202002090000",out_d)

  code = "55022"
  print(code2name(code))
  
  # 11001	11011	11012	11013	11016	11046	11061	11076

  # data_list = getd_110570("11016",202002280000)
  # print(data_list)

  # get_100571("44132","202002090000",out_d)

  # path = f"/home/ysorimachi/work/hokuriku/out/smame/ame/{code}_110570.dat"
  # csv_path=f"/home/ysorimachi/work/hokuriku/out/smame/ame_csv/{code}_110570.csv"
  # open_110570(code,path,csv_path)