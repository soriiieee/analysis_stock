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
import seaborn as sns
#---------------------------------------------------
# import subprocess
# outd ='/home/griduser/work/sori-py2/deep/out/0924_01'
# os.makedirs(outd, exist_ok=True)
# subprocess.run('rm -f *.png', cwd=outd, shell=True)
# #---------------------------------------------------------------------------
# #---------------------------------------------------------------------------
# initial
#

import requests
import json

# from getLonLatGeoCoding import get_latlon #(address)
def get_latlon(address):
  if address is None:
    print("please input geocoding address...")
    sys.exit()
  url = "https://google-maps-geocoding.p.rapidapi.com/geocode/json"
  querystring = {"language":"ja","address":address}

  headers = {
      'x-rapidapi-host': "google-maps-geocoding.p.rapidapi.com",
      'x-rapidapi-key': "81b59639c3msh2b8f90ab786d1b1p191ae4jsnd5ab2653a132"
      }

  res = requests.request("GET", url, headers=headers, params=querystring)
  if res.status_code==200:
    data = res.json()
    # address = data["results"][0]["geometry"]["bounds"]
    address = data["results"][0]["geometry"]["location"]
    return address["lat"], address["lng"]
  else:
    print(f"{address} is None...")
    return 9999., 9999.


if __name__ =="__main__":
  lat,lon = get_latlon(address="長野県須坂市")
  print(f"{lat},{lon}")