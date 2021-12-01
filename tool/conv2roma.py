# -*- coding: utf-8 -*-
# """
# Created on Tue Oct  2 10:04:51 2018
 
# @author: sakamoto
# """
 
import pandas as pd
import os
import sys
import numpy as np
from datetime import datetime

sys.path.append("/home/ysorimachi/.conda/envs/sori_conda/lib/python3.7/site-packages")
from pykakasi import kakasi
import mojimoji

kakasi = kakasi()
kakasi.setMode('H', 'a')
kakasi.setMode('K', 'a')
kakasi.setMode('J', 'a')

conv = kakasi.getConverter()
#---------------------------------------------------

def conv2roma(x, han=False):
  if han:
    x2 = mojimoji.han_to_zen(x)
    y = conv.do(x2)
    return y.upper()
  else:
    y = conv.do(x)
    return y.upper()
    # tbl_ame["name_kana"] = tbl_ame["name_kana"].apply(lambda x: mojimoji.han_to_zen(x))
    # tbl_ame["name_kana"] = tbl_ame["name_kana"].apply(lambda x: conv2romaji(x).upper())


if __name__ == "__main__":
  # DIR_TBL="/home/griduser/work/snowdepth_calc200525/tbl/amedas"
  DIR="/home/griduser/work/sori-py2/timeWeather/tbl"
  input_path=f"{DIR}/ame_master.csv" 
  if os.path.exists(input_path):
    tbl_ame = pd.read_csv(input_path)
    tbl_ame = tbl_ame[['観測所番号', '種類', 'ｶﾀｶﾅ名', '所在地', '緯度(度)', '緯度(分)','経度(度)', '経度(分)']]
    
    #convert
    tbl_ame["name"] = tbl_ame["ｶﾀｶﾅ名"].apply(lambda x: conv2romaji(x, han=True))


    print(tbl_ame.head())
    print(tbl_ame.columns)
    sys.exit()

  # tbl_ame_col0 = ['ff', 'code_ame', 'cate', 'name_kanji', 'name_kana', 'address','lat0', 'lat1', 'lon0', 'lon1', 'height', 'wind_h', 'temp_h', 'start','other', 'other2', 'lon', 'lat', 'other3', 'dx2400', 'dy3600','other4', 'dx480', 'dy720']
  



