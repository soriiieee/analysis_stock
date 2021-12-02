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
# from getErrorValues import me,rmse,mae,r2 #(x,y)
# from convSokuhouData import conv_sfc #(df, ave=minutes,hour)
# #amedas relate 2020.02,04 making...
# from tool_AMeDaS import code2name, name2code
# from tool_110570 import get_110570,open_110570
# from tool_100571 import get_100571,open_100571
# #(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
import subprocess
import pickle
# outd ='/home/griduser/work/sori-py2/deep/out/0924_01'
# os.makedirs(outd, exist_ok=True)
# subprocess.run('rm -f *.png', cwd=outd, shell=True)


#------------2021.10.08 --------------
def save_numpy(save_path,obj):
  save_path2 = save_path.split(".")[0]
  np.save(save_path2, obj.astype('float32'))
  return 

def load_numpy(path):
  obj = np.load(path)
  return obj

def load_numpy2(path):
  obj = np.load(path)
  N,H,W = obj.shape
  obj = obj.reshape(H,W,N) #image　読み込みの為に順番変更 2021.10.26
  return obj

def save_model(path,model):
  with open(path,"wb") as pkl:
    pickle.dump(model,pkl)
  return

def load_model(path):
  with open(path,"rb") as pkl:
    model = pickle.load(pkl)
  return model
