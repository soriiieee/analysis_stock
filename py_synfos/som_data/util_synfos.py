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
# sori -module
sys.path.append('/home/ysorimachi/tool')
from utils_log import log_write #(path,mes,init=False)
#---------------------------------------------------
import subprocess
from tool_time import dtinc
from PIL import Image

from a01_99_utils import *

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    from torch.utils.data import DataLoader, TensorDataset, Dataset
    from torchvision import transforms
    from torchinfo import summary #torchinfoはニューラルネットの中身を見れるのでおすすめ
except:
    print("Not Found Troch Modules ...")

DHOME="/work2/ysorimachi/mix_som/dat" #synfos/ data 

OHOME="/work2/ysorimachi/mix_som/out/syn_data/data"
SYNFOS_INIT = "../../src/now_jst_som.txt"

def synfos_inij():
  with open(SYNFOS_INIT,"r") as f:
    ini_u = f.read()[:12]
  return ini_u

def get_cate():
  """[summary]
  取得したmeshデータから、抽出対象のcateを返却する
  Returns:
      [type]: [description]
  """
  #　指定無し(データ作成用) ----------------
  _f = os.listdir(DHOME)
  _cate = np.unique([ c.split("_")[1][1:] for c in _f])
  #　指定あり(解析用) ----------------
  # _cate = ['70RH','70UU','70VV','85RH','HICA','LOCA','MICA','MSPP']
  # _cate = ['30RH','50RH','70OO','70RH','70UU','70VV','85OO','85RH','85UU','85VV','HICA','LOCA','MICA','MSPP']
  return _cate


def clensing(img,cate):
  """[summary] 2021.11.11
  pnginput　byte data -> value
  Args:
      img ([numpy]): [description]
      cate ([string 4character]): height2 + element2

  Returns:
      [numpy]: [clensing data]
  """
  img = np.flipud(img)
  img = np.where(img ==65535,np.nan,img)
  
  if cate[2:]=="RH":
      pass
  elif cate[2:] == "CA":
      img /= 10
  elif cate[2:] == "UU" or cate[2:] == "VV":
      img = (img- 32768) / 10
  elif cate[2:] == "OO":
      img = (img- 32768) / 1000
  if cate == 'MSPP':
      img /= 10    
  return img

def def_dl(cate):
  if cate[2:] == "OO":
    return 2
  else:
    if cate[:2] == '85' or cate[:2]=="LO" or cate[:2]=="MS":
      return 2
    elif cate[:2]=="70" or cate[:2]=="MI":
      return 5
    elif cate[:2]=="50":
      return 7
    elif cate[:2]=="HI" or cate[:2]=="30":
      return 10
    else:
      return 2
