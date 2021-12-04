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
# sys.path.append('/home/griduser/tool')
# from getErrorValues import me,rmse,mae,r2 #(x,y)
import requests

def get_prod(mode,prd):
  if mode=="UNYO":
    return get_unyo_prod(prd)
  elif mode=="RETRY":
    return get_choki_prod(prd)

def get_unyo_prod(prd):
  id1 = prd.split("-")[0]
  id2 = prd.split("-")[1]
  id3 = prd.split("-")[2]
  init_utc = prd.split("-")[3].split(".")[0]
  # print(id1,id2,id3,init_utc)
  # sys.exit()


  url=f"http://micproxy1.core.micos.jp/product/data/{id1}/{init_utc[0:8]}/{id2}/{id3}/{prd}"
  # url = f"http://micproxy1.core.micos.jp/stock/{yy}/{mm}/{yy}{mm}{dd}/data/100571/{yy}{mm}{dd}/000
  res = requests.get(url,auth=('micosguest', 'mic6guest'))

  return res.status_code, res

def get_choki_prod(prd):
  id1 = prd.split("-")[0]
  id2 = prd.split("-")[1]
  id3 = prd.split("-")[2]
  init_utc = prd.split("-")[3].split(".")[0]


  url=f"http://micproxy2.core.micos.jp/stock/{init_utc[0:4]}/{init_utc[4:6]}/{init_utc[0:8]}/data/{id1}/{init_utc[0:8]}{id2}/{id3}/{prd}"
  # url = f"http://micproxy1.core.micos.jp/stock/{yy}/{mm}/{yy}{mm}{dd}/data/100571/{yy}{mm}{dd}/000
  res = requests.get(url,auth=('micosguest', 'mic6guest'))
  return res.status_code, res

