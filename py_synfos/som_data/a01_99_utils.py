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
# from tqdm import tqdm
# import seaborn as sns
import pickle
#---------------------------------------------------
# sori -module
sys.path.append('/home/ysorimachi/tool')
#(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import r2_score
mms = MinMaxScaler()
DHOME="/work2/ysorimachi/mix_som/out/syn_mesh/data"

SYNFOS_INIT = "../../src/now_jst_som.txt"
CLUSTER="/home/ysorimachi/data/synfos/som/model/cluster"
SOM_DIR="/home/ysorimachi/data/synfos/som/model/labels_synfos/train"



def down_sample(data):
  ny1,nx1 = data.shape
  # dy,dx =int(np.round(ny1/nxny)),int(np.round(nx1/nxny))
  dy,dx =int(np.floor(ny1/nxny)),int(np.floor(nx1/nxny))
  # print(ny1,nx1,dy,dx)
  # sys.exit()
  data2 = np.zeros((nxny,nxny))
  for i in range(nxny):
    for j in range(nxny):
      if j == nxny-1:
        v = data[i*dy:(i+1)*dy,j*dx:]
      else:
        v = data[i*dy:(i+1)*dy,j*dx:(j+1)*dx]
      data2[i,j] = np.nanmean(v)
      # print(i*dy,(i+1)*dy,j*dx,(j+1)*dx, "v=",np.nanmean(v))
      # print(i*dy,(i+1)*dy,j*dx,(j+1)*dx, "v=",v)
  # print(ny1,nx1,dy,dx)
  return data2

def trim_idx(data,lonlat=[125,145,25,45]):
  ny,nx = data.shape
  _lon = np.linspace(120,150,nx)
  _lat = np.linspace(20,50,ny)
  
  x0,x1,y0,y1 = lonlat
  
  iy0 = np.argmin(np.abs(_lat - y0))
  iy1 = np.argmin(np.abs(_lat - y1))
  ix0 = np.argmin(np.abs(_lon - x0))
  ix1 = np.argmin(np.abs(_lon - x1))
  return iy0,iy1,ix0,ix1
  

#-------------------------------

def main():
  v_hash={}
  for (i,ini_j) in tqdm(list(enumerate(_jst))):
    path = f"{MESH}/{ee}_{ini_j}_ft1.csv"
    
    if os.path.exists(path):
      df = pd.read_csv(path)
      
      #2021.07.07　-確認-
      # print(np.nanmin(df.values), np.nanmax(df.values))
      # print(df.shape)
      # sys.exit()

      data = np.array(df.values)
      # print(data.shape)
      # sys.exit()
      ilon,ilat = int(lon),int(lat)
      ilon,ilat = float(lon),float(lat)
      # print(ilon,ilat)
      # sys.exit()
      """
      気象要素別に切り出しエリアの調整
      """
      # ee = "LOCA"
      if ee == "SFTT":
        lonlat=[ilon-12,ilon+6,ilat-10,ilat+8]
      elif ee== "MSPP":
        lonlat=[ilon-12,ilon+6,ilat-10,ilat+8]
      elif ee== "HICA":
        lonlat=[ilon-12,ilon+6,ilat-10,ilat+8]
      elif ee== "MICA":
        lonlat=[ilon-5,ilon+5,ilat-5,ilat+5]
      elif ee== "LOCA":
        lonlat=[ilon-1,ilon+1,ilat-1,ilat+1]
        
      y0,y1,x0,x1 = trim_idx(data,lonlat=lonlat)
      data = data[y0:y1,x0:x1]
  
  return data


def load_10():
  # path = "../../tbl/list_10.tbl"
  path = "/home/ysorimachi/work/ecmwf/tbl/list_10.tbl"
  df = pd.read_csv(path , delim_whitespace=True,header=None)
  df=df[[0,1,2,27]]
  # df = df.set_index(0)
  # name_dict = df.to_dict()[27]
  # return name_dict
  return df


def isFloat(x):
  try:
    return float(x)
  except:
    return np.nan



def synfos_inij():
    with open(SYNFOS_INIT,"r") as f:
        ini_u = f.read()[:12]
    return ini_u

def get_cate():
    _f = os.listdir(DHOME)
    _cate = np.unique([ c.split("_")[1][1:] for c in _f])
    # _cate = ['70RH','70UU','70VV','85RH','HICA','LOCA','MICA','MSPP']
    return _cate

def check():
    _f= sorted(glob.glob(f"{DHOME}/*_070RH*.png"))
    print(_f)
    sys.exit()


def load_img(cate,hh):
    path = f"{DHOME}/wrf_0{cate}_12_{hh}.png"
    img = np.array(Image.open(path))
    img = clensing(img,cate)
    return img

#------------2021.10.08 --------------
def save_numpy(save_path,obj):
    save_path2 = save_path.split(".")[0]
    np.save(save_path2, obj.astype('float32'))
    return 
def load_numpy(path):
    obj = np.load(path)
    return obj
def save_model(path,model):
  with open(path,"wb") as pkl:
    pickle.dump(model,pkl)
  return
def load_model(path):
  with open(path,"rb") as pkl:
    model = pickle.load(pkl)
  return model
#------------2021.10.08 --------------


def loop_iniu():
    _t = pd.date_range(start= "201903302100",freq="1D", periods=735)
    _t = [t.strftime("%Y%m%d%H%M") for t in _t]
    # print(_t)
    return _t

def get_mesh_u(ini_u0):
    fd_time = dtinc(ini_u0,4,39)
    return fd_time


def load_som_model(cate,ndim,isCNN):
    if cate=="MSPP":
        cc = "sp"
    if cate=="LOCA":
        cc = "lc"
    if cate=="MICA":
        cc = "mc"
    if cate=="HICA":
        cc = "hc"
        
    path = f"{CLUSTER}/{cc}_{ndim}_{isCNN}.pkl"
    model = load_model(path)
    return model

def train_flg(df):
  if df["time"].dtypes == object:
    df["time"] = pd.to_datetime(df["time"])
  df["dd"] = df["time"].apply(lambda x: x.day)
  # df["istrain"] = df["dd"].apply(lambda x: 0 if x%5==4 else 1)
  df["istrain"] = df["dd"].apply(lambda x: 0 if x%2==1 else 1)
  df = df.drop(["dd"],axis=1)
  return df
  
def train_flg2(df):
    df["dd2"] = df["dd"].apply(lambda x: int(x[4:6]))
    # df["istrain"] = df["dd2"].apply(lambda x: 0 if x%5==4 else 1)
    df["istrain"] = df["dd2"].apply(lambda x: 0 if x%2==1 else 1)
    df = df.drop(["dd2"],axis=1)
    return df


def select_som_day(cele,lbl=0,istrain=True,istest=False):
    # k-meansで分類された16分類に該当する、64分類のsomクラスタに分類された日にちを取得して学習モデルとして実施する
    def list_som_dd():
        path = f"{SOM_DIR}/label_som.csv"
        df = pd.read_csv(path)
        if istrain:
            df = df[df["istrain"]==1]
        else:
            if istest:
                df = df[df["istrain"]==0]
            else:
                pass
            
        df["dd"] = df["dd"].apply(lambda x: str(x)[:8])
        # df = df.loc[df[cele].isin(_cluster),:]
        df = df[df[cele]==lbl]
        if not df.empty:
            return df["dd"].values.tolist()
        else:
            return []
    _dd = list_som_dd()
    return len(_dd),_dd
  

def kaizen_rate(df,c1,c2):
  
  # df = df.replace(9999,9999)
  df = df.replace(0,9999)
  df[c1] = np.abs(df[c1])
  df[c2] = np.abs(df[c2])
  
  #------------------
  def calc_rate(x):
    
    if x[0] ==9999. or x[1]==9999.:
      return 9999
    else:
      v = -1 * (x[1]-x[0])/x[0] *100
      return v
  #------------------
  df["rate"] = df[[c1,c2]].apply(lambda x:calc_rate(x),axis=1)
  list_v = df["rate"].values
  return list_v


def calc_ci_day(df,path=None):
  _dd = sorted(df["dd"].unique().tolist())
  _mean,_std,_coef,_r2=[],[],[],[]
  mms = MinMaxScaler()
  
  for i,dd in enumerate(_dd):
    # dd="20190513"
    tmp = df[df["dd"]==dd]
    tmp = tmp.dropna(subset=["OBS","CR0"])
    if tmp.shape[0] !=0:
      tmp["ci"] = tmp["OBS"]/ tmp["CR0"]
      mean = np.mean(tmp["ci"])
      std = np.std(tmp["ci"])
      X = tmp[["OBS","CR0"]].values
      coef = np.corrcoef(X[:,0],X[:,1])[0,1]
      r2 = r2_score(X[:,0],X[:,1])
    else:
      mean,std,coef,r2 = 9999,9999,9999,9999
    
    if 0:
      plt.scatter(X[:,0],X[:,1])
      plt.savefig("./sample.png")
      sys.exit()
    # print(X)
    # print(X.shape)
    # print(coef,r2)
    # sys.exit()
    if i %100==0:
      print(datetime.now(),"[END]--> NUM=",i, np.round(mean,2),np.round(std,2),np.round(coef,2))
      pass
    _mean.append(mean)
    _std.append(std)
    _coef.append(coef)
    _r2.append(r2)
  
  
  df = pd.DataFrame()
  df["dd"] = _dd
  df["mean"] = _mean
  df["std"] = _std
  df["coef"] = _coef
  df["r2"] = _r2
  
  # df = df.set_index("dd")
  if path:
    df.to_csv(path,index=False)
  else:
    return df

def sub_plot_bar(ax,df):
    w = np.round(1.0/(len(df.columns)+1),2)
    _index = df.index
    """
    ax : matplotlib
    df : index/cは表示するカラムのみ
    """ 
    for i,c in enumerate(df.columns):
      ax.bar(w/2 + np.arange(len(df))+w*i,df[c],width=w,label=c,align="edge")
    
    ax.set_xlim(0,len(df))
    ax.set_xticks(np.arange(len(df))+1/2)
    for x0 in np.arange(len(df)):
      ax.axvline(x=x0,color="gray", alpha=0.5, lw=1)
      
    ax.set_xticks(np.arange(len(df))+1/2)
    ax.set_xticklabels(_index,rotation=0)
    return ax

def autolabel(rect,ax):
  """Attach a text label above each bar in *rects*, displaying its height."""
  for rect in rects:
    height = rect.get_height()
    ax.annotate('{}'.format(height),xy=(rect.get_x() + rect.get_width() / 2, height),xytext=(0, 3),  # 3 points vertical offsetextcoords="offset points",
    ha='center', va='bottom')
  return ax


def cut_time(df,st,ed):
  if not "hh" in df.columns:
    df["hh"] = df["time"].apply(lambda x: int(x.strftime("%H")))
  df = df[(df["hh"]>=st)&(df["hh"]<=ed)]
  df = df.drop(["hh"],axis=1)
  return df
   

if __name__ == "__main__":
  load_weather_fcs()