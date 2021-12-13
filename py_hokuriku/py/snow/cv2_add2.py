# -*- coding: utf-8 -*-
# when   : 2020.0x.xx
# who : [sori-machi]
# what : [ ]
#---------------------------------------------------------------------------
# basic-module
import matplotlib.pyplot as plt
import sys,os,re,glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.simplefilter('ignore')
from tqdm import tqdm
import seaborn as sns
#---------------------------------------------------
import cv2
# print(cv2.__file__)
# sys.exit()
"""
画像を重ねるチュートリアル
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_core/py_image_arithmetics/py_image_arithmetics.html#bitwise-operations

"""

def main():
  im_p = "/home/ysorimachi/work/hokuriku/out/snow/111095_202002010000.png"
  m_p = "/home/ysorimachi/work/hokuriku/out/snow/mask/hokuriku_area.png"
  out_p = "/home/ysorimachi/work/hokuriku/out/snow/concat/111095_202002010000.png"
  
  img = cv2.imread(im_p)
  mask = cv2.imread(m_p)
  
  print(img.shape)
  print(mask.shape)
  return

if __name__ =="__main__":
  main()