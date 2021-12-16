# -*- coding: utf-8 -*-
# when   : 2021.04.12
#---------------------------------------------------------------------------
# basic-module
import sys,os,re,glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.simplefilter('ignore')
from tqdm import tqdm
#---------------------------------------------------
# sori -module
sys.path.append('/home/ysorimachi/tool')

#(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
import subprocess
import xml.etree.ElementTree as ET

DHOME="/home/ysorimachi/work/ecmwf/csv"
DHOME2="/home/ysorimachi/work/ecmwf/csv2"
DHOME3="/home/ysorimachi/work/ecmwf/csv3"

power_dict ={
  0:"hkdn", #北海道
  1:"thdn", #東北
  2:"tden", #東京電力
  3:"hrdn",#北陸
  4:"cbdn",#中部
  5:"kndn",#関西
  6:"cgdn",#中国
  7:"yndn",#四国
  8:"kypv",#九州
  9:"okdn",#沖縄
}


imiss,rmiss = 9999,9999.

def _ec(ini_ec,n,cate="ens"):
  #---------------------
  if cate =="ens":
    path = f"{DHOME}/117012-410{n}06-8999-{ini_ec}00.xml"
  else:
    path = f"{DHOME}/117012-410{n}06-8000-{ini_ec}00.xml"
  #---------------------
  id3 = os.path.basename(path).split("-")[2]
  with open(path,"r") as f:
    data = f.read()
  
  root = ET.fromstring(data)
  for pnt in root.findall("point"):
    code = pnt.attrib["code"]
    
    _rad,_ft=[],[]
    df = pd.DataFrame()
    for fct in pnt.findall("forecast"):
      rad = fct.find("solarRad").text
      if rad is None:
        rad = imiss
      ft = fct.attrib["time"]
      if ft is None:
        ft = imiss
      else:
        ft = ft[:16] +":00"
      _rad.append(rad)
      _ft.append(ft)
    
    df["ft"] = _ft
    df["rad"] = _rad
    df.to_csv(f"{DHOME2}/{code}_{id3}.csv", index=False)
  return

def power_name(code):
  n = int(code[3:4])
  return power_dict[n]

def power_n(code):
  n = code[-3:]
  return n.zfill(3)


def concat(ini_ec):
  # cleaning...
  subprocess.run("rm -rf *.csv", cwd=DHOME3, shell=True)
  #concat
  _code = np.unique([ fname[:9] for fname in  os.listdir(DHOME2)])
  for code in tqdm(_code):
    p_name = power_name(code)
    code2 = power_name(code) + power_n(code)
    
    p1 = f"{DHOME2}/{code}_8999.csv"
    p2 = f"{DHOME2}/{code}_8000.csv"
    if os.path.exists(p1) and os.path.exists(p2):
      df1 = pd.read_csv(p1)
      df2 = pd.read_csv(p2)
      df = df1.merge(df2, on="ft",how="inner")
      
      fname = f"{code2}_{ini_ec}_ecm.csv"
      df.to_csv(f"{DHOME3}/{fname}",index=False,header=None)
    else:
      pass


if __name__=="__main__":
  # ini_ec="202012090000"
  ini_ec = sys.argv[1]
  #init
  # subprocess.run("rm -rf *.csv", cwd=DHOME2, shell=True)
  for n in tqdm([2,3,4,5,6,7,8,9]):
    for cate in ["ens","cc"]:
      _ec(ini_ec,n,cate=cate)
  #init2
  concat(ini_ec)
  
