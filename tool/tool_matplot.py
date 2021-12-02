# -*- coding: utf-8 -*-
# when   : 2020.0x.xx
# who : [sori-machi]
# what : [ ]
#---------------------------------------------------------------------------
# basic-module
import matplotlib.pyplot as plt
import sys,os,re,glob
import pandas as pd
import subprocess
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.simplefilter('ignore')


import matplotlib
from matplotlib import font_manager

def set_japanese(fontsize=12):
  # caches remove / ------------------
  subprocess.run("rm ~/.cache/matplotlib/fontlist-v330.json",shell=True)
  subprocess.run("rm ~/.cache/matplotlib/fontlist-v310.json",shell=True)
#     matplotlib.font_manager._rebuild()

  from matplotlib import rcParams
  import japanize_matplotlib  # <- これ
  rcParams['font.family'] ='IPAexGothic'
  rcParams['font.size'] = fontsize
  rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
# set_japanese_matplot()
# set_japanese_matplot()
# https://matplotlib.org/stable/api/font_manager_api.html
# font_manager.FontManager.addfont("/home/ysorimachi/.local/lib/python3.8/site-packages/matplotlib/mpl-data/fonts/ttf/ipaexg.ttf")

def check_font():
  from matplotlib import font_manager
  for i in font_manager.fontManager.ttflist:
    if ".tt" in i.fname:
      print(i) 
#     font_manager.findSystemFonts()
    
# check_font()
# set_japanese_matplot()
# font_manager.findSystemFonts()

if __name__ == "__main__":
  
  set_japanese(fontsize=12)
  print("matplotlib_fname",matplotlib.matplotlib_fname())
  print("get_configdir",matplotlib.get_configdir())
  # MATPLOTDIR
  # /home/ysorimachi/.local/lib/python3.8/site-packages/matplotlib/mpl-data/fonts/ttf　以下に保存
  check_font()