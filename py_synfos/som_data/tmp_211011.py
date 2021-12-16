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
from getErrorValues import me,rmse,mae,r2 #(x,y)
# from convSokuhouData import conv_sfc #(df, ave=minutes,hour)
# #amedas relate 2020.02,04 making...
# from tool_AMeDaS import code2name, name2code
# from tool_110570 import get_110570,open_110570
# from tool_100571 import get_100571,open_100571
#(code,ini_j,out_d)/(code,path,csv_path)

from utils_log import log_write
#---------------------------------------------------
import subprocess

import netCDF4
from netCDF4 import Dataset
import pickle 
from PIL import Image
from util_Layers import CNN_FOR_SOM, Scaler
from util_SOM2 import ClusterModel

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from torchvision import transforms
from torchinfo import summary #torchinfoはニューラルネットの中身を見れるのでおすすめ

class OpenNetCDF:
    DIR_DATA="/work/ysorimachi/era5/dat2"
    
    def __init__(self):
        self.cele = None
        self.mm = None
        self.nc = None
        
    def path(self):
    #     high_cloud_cover  low_cloud_cover  mean_sea_level_pressure  medium_cloud_cover
        if self.cele == "hc":
            name = "high_cloud_cover"
        elif self.cele == "mc":
            name = "medium_cloud_cover"
        elif self.cele == "lc":
            name = "low_cloud_cover"
        elif self.cele == "sp":
            name = "mean_sea_level_pressure"
            
        path = f"{self.DIR_DATA}/{name}/download_{self.mm}.nc"
        return path
    
    def load(self,cele,mm):
        self.cele = cele
        self.mm =mm
        
        path = self.path()
        if path is not None:
            nc = Dataset(path, 'r')
            self.nc = nc
            
            data = self._data
            if type(data) == netCDF4._netCDF4.Variable:
                data = np.array(data)
            time = self._time
            _lat = self._lat
            _lon = self._lon
            return data,time,_lat,_lon
            
        else:
            print("Not Found !")
    
    def load_multi(self,cele,_mm):
        _data =[]
        _time = []
        for mm in tqdm(_mm):
            data,time,_lat,_lon = self.load(cele,mm)
            _time += time
            _data.append(data)
        
        data = np.concatenate(_data)
        return data,_time,_lat,_lon
    
#     @property
    def check(self,cele,mm):
        self.cele = cele
        self.mm =mm
        path = self.path()
        if path is not None:
            nc = Dataset(path, 'r')
        return nc.variables.keys()
    
    @property
    def _data(self):
        if self.cele == "hc":
            k = "hcc"
        elif self.cele == "mc":
            k = "mcc"
        elif self.cele == "lc":
            k = "lcc"
        elif self.cele == "sp":
            k = "msl"
        return self.nc.variables[k]
    
    @property
    def _time(self):
        _t = list(self.nc.variables["time"])
        _t = [ self.conv_time(t) for t in _t ]
        return _t
    
    def conv_time(self,t):
        init = datetime(1900,1,1,0,0)
        return init + timedelta(hours=int(t.data))
    
    @property
    def _lon(self):
        _lon = [ float(t.data) for t in self.nc.variables["longitude"] ]
        return _lon
    @property
    def _lat(self):
        _lat = [ float(t.data) for t in self.nc.variables["latitude"] ]
        return _lat

def loop_mm(st=2001,ed=2020):
  _mm=[]
  for yy in range(st,ed+1):
    for mm in range(1,12+1):
      cmm= str(mm).zfill(2)
      _mm.append(f"{yy}{cmm}")
  print("N_month ->", len(_mm))
  return _mm


def preprocess(img):
  pre_prs = transforms.Compose([
    transforms.Resize(56),
    # transforms.CenterCrop(56),
    transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         transforms.Grayscale(num_output_channels=1),
    ])
  img = pre_prs(img)
    # Min -Max Scaler
#     https://discuss.pytorch.org/t/how-to-efficiently-normalize-a-batch-of-tensor-to-0-1/65122
  return img

def preprocess_0(img):
  pre_prs = transforms.Compose([
    transforms.Resize(14),
    # transforms.CenterCrop(56),
    transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         transforms.Grayscale(num_output_channels=1),
    ])
  img = pre_prs(img)
    # Min -Max Scaler
#     https://discuss.pytorch.org/t/how-to-efficiently-normalize-a-batch-of-tensor-to-0-1/65122
  return img


def mk_DataLoader(data,_time, isCNN):
  sc = Scaler("minmax")
  
  data = sc.fit_transform(data)
  
  if type(data) != np.ndarray:
    data = np.array(data)
  
  _img=[]
  for i in range(data.shape[0]):
    img = data[i,:,:]
      
    if type(img) != Image.Image:
      img = Image.fromarray(img)
      
    if isCNN:
      img = preprocess(img)
    else:
      img = preprocess_0(img) #かなり画像解像度を下げる
    _img.append(img)
  
  _img = tuple(_img)
  _img = torch.stack(_img,0)
  return _img,_time

# _img,_time = mk_DataLoader("sp",_
def train_cleaner():
  subprocess.run(f"rm {CLUSTER}/*.pkl",shell=True)
  subprocess.run(f"rm {LABELS}/*.csv",shell=True)
  subprocess.run(f"rm {MESH}/*.png",shell=True)
  print("all director Cleaned !")
  return

CLUSTER="/home/ysorimachi/data/synfos/som/model/cluster"
LABELS="/home/ysorimachi/data/synfos/som/model/labels"
MESH="/home/ysorimachi/data/synfos/som/model/mesh"
def train_SOM_cluster(cele="sp",n_month=12,_n_dim=[6],_isCNN=[0]):
  nc = OpenNetCDF()
  _mm = loop_mm(st=2001,ed=2020)[:n_month]
  # CNN---------
  
  data,_time,_,_ = nc.load_multi(cele,_mm)
  data = data[4::8,:,:] #12時時点のデータのみ利用する
  _time = _time[4::8] #12時時点のデータのみ利用する
  
  # som model save ---
  cm_path = f"{CLUSTER}/{cele}_{n_month}_{n_dim}_{isCNN}.pkl"
  som_model = load_model(cm_path)
  
  print(som_model)
  sys.exit()
  
# loop 
  for n_dim in _n_dim: #4/8/10
    for isCNN in _isCNN: #0/1
      img,list_t = mk_DataLoader(data,_time,isCNN)
      if isCNN:
        model = load_cnn() #保存したcnnを取得する
        img2 = model(img) # torch.Size([365, 128]) 特徴量の抽出
        img2 = img2.detach().numpy()
      else:
        img2 = img.detach().numpy()
        N,D,nx,ny = img2.shape
        img2 = img2.reshape(N,nx*ny)

      DIMENTION=img2.shape[1]
      # CNN---------
      
      #som trainer ---
      cm = ClusterModel("SOM-ED",n_clusters=n_dim*n_dim,DIMENTION=DIMENTION)
      cm.fit(img2)
      
      # som model save ---
      cm_path = f"{CLUSTER}/{cele}_{n_month}_{n_dim}_{isCNN}.pkl"
      save_model(cm_path,cm)
      # som label save ---
      df = pd.DataFrame()
      df["time"] = _time
      df["label"] = cm.labels_
      df.to_csv(f"{LABELS}/{cele}_{n_month}_{n_dim}_{isCNN}.csv", index=False)
      # som mesh save ---
      f,ax = plt.subplots(,figsize=(25,25))
      ax = ax.flatten()
      N_ALL = cm.labels_.shape[0]
      for n in range(n_dim*n_dim):
        sub = data[cm.labels_==n,:,:]
        N=sub.shape[0]
        p = np.round(N*100/N_ALL,1)
        if sub.shape[0] != 0:
          center = np.mean(sub,axis=0)
          center -= np.mean(center)
          # print(np.nanmin(center), np.nanmax(center))
          # sys.exit()
          # print(center.shape)
          # sys.exit()
          ax[n].imshow(center,cmap="seismic",vmin=-1,vmax=1)
          ax[n].set_title(f"cls{n}(N:{N}/{p}%)")
          ax[n].tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
        else:
          ax[n].set_visible(False)

      
      plt.subplots_adjust(wspace=0.2, hspace=0.2)
      f.savefig(f"{MESH}/{cele}_{n_month}_{n_dim}_{isCNN}.png", bbox_inches="tight")
      plt.close()
      
      log_write("./c01_train.log",f"nmonth{n_month} {cele} ndim{n_dim} isCNN{isCNN}")
  return 

def load_cnn():
  path = "/home/ysorimachi/work/togo_fcs/model/cnn/2D56_to_1D128.pkl"
  if not os.path.exists(path):
    print("NEW CNN MODLE!")
    model = CNN_FOR_SOM()
    with open(path,"wb") as pkl:
      pickle.dump(model,pkl)
    return model
  else:
    print("ALREADY GET!")
    with open(path,"rb") as pkl:
      model = pickle.load(pkl)
    return model
  
def save_model(path,model):
  with open(path,"wb") as pkl:
    pickle.dump(model,pkl)
  return

def load_model(path):
  with open(path,"rb") as pkl:
    model = pickle.load(pkl)
  return model



if __name__== "__main__":
  
  
  #--------------------
  if 0:
    if 0:
      train_cleaner()
      log_write("./c01_train.log","start",init=True)
    # sys.exit()
    n_month=12*15
    _n_dim=[8] #som map size
    _isCNN =[0,1]
    for cele in ["sp","mc","lc","hc"]: #"sp","mc","lc","hc"
      train_SOM_cluster(cele=cele,n_month=n_month,_n_dim=_n_dim,_isCNN=_isCNN) #2021.10.05
      
      
          
          
          
