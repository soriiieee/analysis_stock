# -*- coding: utf-8 -*-
# when   : 2021.11.10
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
# import seaborn as sns
#---------------------------------------------------
# sori -module
sys.path.append('/home/ysorimachi/tool')
from getErrorValues import me,rmse,mae,r2 #(x,y)
from convSokuhouData import conv_sfc #(df, ave=minutes,hour)
#amedas relate 2020.02,04 making...
from tool_AMeDaS import code2name, name2code
from tool_110570 import get_110570,open_110570
from tool_100571 import get_100571,open_100571
#(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
import subprocess
# outd ='/home/griduser/work/sori-py2/deep/out/0924_01'
# os.makedirs(outd, exist_ok=True)
# subprocess.run('rm -f *.png', cwd=outd, shell=True)

#------------------
# png 画像のフォーマットについて調べるような作業
# https://water2litter.net/rum/post/python_pil_image_attributes/
from PIL import Image
from io import BytesIO 

from pngparser import PngParser, ImageData, TYPE_IHDR
import sys


def load_img(T=1):
  path = f"/home/ysorimachi/data/synfos/tmp/gsm/check/{T}/jprhsf12087.png"
  img = Image.open(path)
  return img

def img_path(T=1):
  return f"/home/ysorimachi/data/synfos/tmp/gsm/check/{T}/jprhsf12087.png"

def load_img2(T=1):
  path = f"/home/ysorimachi/data/synfos/tmp/gsm/check/{T}/jprhsf12087.png"
  with open(path, 'rb') as f:
    binary = f.read()
  
  with open(f"./binary-{T}.dat", "w") as f:
    f.write(str(BytesIO(binary).getvalue()))
    
  img = Image.open(BytesIO(binary))
  return img

def filesize(T=1):
  path = f"/home/ysorimachi/data/synfos/tmp/gsm/check/{T}/jprhsf12087.png"
  size = os.path.getsize(path)
  return size


def check():
  im1 = load_img2(T=1)
  im2 = load_img2(T=6)
  sys.exit()
  print(filesize(1), filesize(6))
  print(im1.size, im2.size)
  print(im1.format, im2.format)
  print(im1.mode, im2.mode) #I: 32bit(4byte 符号付整数)
  print(im1.info)
  print(im2.info)



def extractData(png):

    header = png.get_by_type(TYPE_IHDR)[0]

    img = png.get_image_data()

    data_bin = ""
    data = ""
    for sc in img.scanlines:
      data += str(sc.filter)

      # hidden challenge
      d = sc.filter % 4
      f = "{:02b}".format(d)
      data_bin += f

    print(f"[*] filter {data}\n")

    data = b''.join(int(data_bin[i:i+8], 2).to_bytes(1, 'big')
                    for i in range(0, len(data_bin), 8))
    print(data)



def main():
  png1 = PngParser(img_path(T=1))
  png2 = PngParser(img_path(T=6))
  print(img_path(T=1))
  # extractData(png1)
  # extractData(png2)
  
  

if __name__ == "__main__":
  # check()
  main()