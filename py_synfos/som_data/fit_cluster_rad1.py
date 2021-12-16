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
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
mms = MinMaxScaler()

from a01_99_utils import *
from c01_som_cluster import *
from util_Data import load_rad, load_10, load_weather_fcs
from util_Model2 import Resid2
import pickle

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


# DHOME="/work2/ysorimachi/mix_som/dat" #synfos/ data 

# OHOME="/work2/ysorimachi/mix_som/out/syn_data/data"
# OHOME="/work2/ysorimachi/mix_som/out/syn_mesh/data"
# DHOME="/work2/ysorimachi/mix_som/out/syn_mesh/data" #2021.10.15
DHOME="/work2/ysorimachi/mix_som/out/syn_data/data"  #2021.11.19

FIT_DIR="/home/ysorimachi/data/synfos/som/model/fitting_synfos"
ESTIMATE1="/home/ysorimachi/data/synfos/som/model/estimate1"
MODEL1="/home/ysorimachi/data/synfos/som/model/m1"
MODEL2="/home/ysorimachi/data/synfos/som/model/m2"

SYNFOS_INIT = "../../src/now_jst_som.txt"
CLUSTER="/home/ysorimachi/data/synfos/som/model/cluster"
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 18
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']


def synfos_inij():
    with open(SYNFOS_INIT,"r") as f:
        ini_u = f.read()[:12]
    return ini_u

def get_cate():
    _f = os.listdir(DHOME)
    # _cate = np.unique([ c.split("_")[1][1:] for c in _f])
    _cate = ['70RH','70UU','70VV','85RH','HICA','LOCA','MICA','MSPP']
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
    #update 2021.11.19
    # _t = pd.date_range(start= "201903302100",freq="1D", periods=735)
    _t = pd.date_range(start= "201804012100",end="202109302100",freq="1D")
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

def convert_fit(img):
    """
    # 2021.10.11
    synfos のメッシュからsom読み込み用に、CNNで特徴量抽出
    """
    img = img.reshape((1,1,56,56))
    img_tensor = torch.from_numpy(img.astype(np.float32)).clone()
    cnn = load_cnn()
    img = cnn(img_tensor)
    img = img.detach().numpy()
    return img

LBL_SYNFOS="/home/ysorimachi/data/synfos/som/model/labels_synfos"
def predict_som_cluster(size,FD,ndim,isCNN=1):
    ###
    # 2021年10月1日　synfosの対象予測時刻のファイルから参照すべき地点の情報を変動について計算していく予定
    ###
    LOG_PATH= "/home/ysorimachi/work/synfos/py/som_data/a01_03.log"
    _t = loop_iniu()
    # print(_t[:3],_t[-3:])
    # sys.exit()
    _ini_j = [get_mesh_u(t) for t in _t]
    _cele=["MSPP","HICA"]
    #file check ---------
    # for ini_j in _ini_j:
    #     ODIR=f"{DHOME}/{ini_j[:8]}" #予測対象初期値(JST)
    #     num = len(os.listdir(ODIR))
    #     if num < 16:
    #         # print
    #         print(ini_j, num)
    # sys.exit()
    #--------------------
    
    # 初期時刻の読込- ---
    # ODIR=f"{OHOME}/{ini_j0}" #予測対象初期値(JST)
    
    hash1 ={}
    for ini_j in tqdm(_ini_j): #synfos 予測初期値(UTC)
        # print(ini_j)
        
        ODIR=f"{DHOME}/{ini_j[:8]}" #予測対象初期値(JST)
        # print(os.listdir(ODIR))
        # sys.exit()
        # if ini_j[:8] != "20200602":
        if len(os.listdir(ODIR)) >0:
            _lbl=[]
            # for cate in ["MSPP","LOCA","MICA","HICA"]:
            for cate in _cele:
                mesh_path = f"{ODIR}/FD{FD}_{cate}_s{size}.npy"
                img = load_numpy(mesh_path) #(56,56)
                img = convert_fit(img) #(128)
                
                #som model
                model = load_som_model(cate,ndim,isCNN=isCNN)
                lbl = model.predict(img).values[0]
                _lbl.append(lbl)
                
            hash1[ini_j] = _lbl
        else:
            print(ini_j[:8], "isnot FILE !")
        # print(datetime.now(),"[END]" , ini_j)
    
    df= pd.DataFrame(hash1).T
    df.index.name = "dd"
    df.columns = _cele
    df = df.reset_index()
    df = train_flg(df)
    df = df.set_index("dd")
    df.to_csv(f"{LBL_SYNFOS}/train/label_som_{ndim}_{isCNN}.csv")
    print(f"{LBL_SYNFOS}/train/label_som_{ndim}_{isCNN}.csv")
    return 

def check():
    DIR="/work2/ysorimachi/mix_som/out/syn_mesh/data/20190331"
    _path = sorted(glob.glob(f"{DIR}/*.npy"))

    f,ax = plt.subplots(4,4,figsize=(20,20))
    ax = ax.flatten()
    for i,path in enumerate(_path):
        img = load_numpy(path)
        name = path.split("/")[-1].split(".")[0]
        
        ax[i].imshow(img,vmin=0,vmax=1, cmap="seismic")
        ax[i].set_title(name)

        ax[i].tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
        plt.subplots_adjust(wspace=0.2, hspace=0.2)
    f.savefig("/home/ysorimachi/data/synfos/som/model/mesh/check_synfos_mesh_data.png", bbox_inches="tight")
    print("end")
    return

def train_flg(df):
    df["istrain"] = df["dd"].apply(lambda x: 0 if float(str(x)[6:8])%5==4 else 1)
    return df


def distribution(cate,map_dim,p=True):
    if cate=="MSPP":
        cc = "sp"
    if cate=="LOCA":
        cc = "lc"
    if cate=="MICA":
        cc = "mc"
    if cate=="HICA":
        cc = "hc"
    #ec ----
    ec = pd.read_csv(f"{LABELS}/{cc}_{map_dim}_1.csv")
    n_ec = ec.shape[0]
    ec = ec.groupby("label").count()
    ec = ec.values.reshape(map_dim,map_dim)
    if p:
        ec = ec *100/n_ec
        ec = ec.round(2)
    
    
    #syn
    syn = pd.read_csv(f"/home/ysorimachi/data/synfos/som/model/labels_synfos/train/label_som.csv")
    syn = train_flg(syn)
    
    syn = syn[syn["istrain"]==1] #学習データのみ
    n_syn = syn.shape[0]
    n_count = {}
    for lbl in range(map_dim*map_dim):
        n = syn[syn[cate]==lbl].shape[0]
        n_count[lbl] = [n]
    
    syn = pd.DataFrame(n_count).T
    syn = syn.values.reshape(map_dim,map_dim)
    if p:
        syn = syn *100/n_syn
        syn = syn.round(2)
        
    f,ax = plt.subplots(1,2,figsize=(20,10))
    if p:
        sns.heatmap(ec,vmin=0,vmax = np.max(ec)+1,annot=True,square=True,cmap="Oranges",ax=ax[0])
        sns.heatmap(syn,vmin=0,vmax= np.max(syn)+1,annot=True,square=True,cmap="Oranges",ax=ax[1])
    else:
        sns.heatmap(ec,vmin=0,vmax = np.max(ec)+1,annot=True,square=True,fmt="d",cmap="Oranges",ax=ax[0])
        sns.heatmap(syn,vmin=0,vmax= np.max(syn)+1,annot=True,square=True,fmt="d",cmap="Oranges",ax=ax[1])
    
    ax[0].set_title(f"ec-era5 10year 分布 cate({cate})", fontsize=12)
    ax[1].set_title(f"synfos 2year 分布 cate({cate})", fontsize=12)
    
    f.savefig(f"/home/ysorimachi/data/synfos/som/model/labels_synfos/train/heatmap_{cate}.png", bbox_inches="tight")
    return

def kmeans_agglo(cate,map_dim,n_clusters,PNG=False):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters)
    cm = load_som_model(cate,map_dim,isCNN=1)
    
    val = cm.cluster_centers_.reshape(-1,128)
    km.fit(val)
    img = km.labels_.reshape(map_dim,map_dim)
    
    index = list(km.labels_)
    if PNG:
        f,ax = plt.subplots(figsize=(10,10))
        sns.heatmap(img,annot=False,square=True,cmap="jet",ax=ax,linewidths=.5)
        ax.set_title(f"KMeans(8*8-> {n_clusters}) -> agglomeration({cate})", fontsize=18)
        f.savefig(f"/home/ysorimachi/data/synfos/som/model/labels_synfos/train/kmenas{n_clusters}_{cate}.png", bbox_inches="tight")
        plt.close()
    return index


def concat_cluster(map_dim = 4,n_clusters=16):
    """
    concat 
    2021.10.12 stat !!
    """
    # hash_res = {}
    for cate in ["MSPP","LOCA","MICA","HICA"]:
        if 1:
            distribution(cate,map_dim,p=False) #synfosの分布(どのカテゴリに多く分類されるのか)
    hash_res = {}
    for cate in ["MSPP","LOCA","MICA","HICA"]:
        _index = kmeans_agglo(cate,map_dim,n_clusters=9,PNG=True)
        hash_res[cate] = _index
        
    df = pd.DataFrame(hash_res)
    df.columns = ["MSPP","LOCA","MICA","HICA"]
    df.index.name = "cluster"
    # df = df.reset_index()
    out_path = "/home/ysorimachi/data/synfos/som/model/labels_synfos/train/label_kmeans.csv"
    df.to_csv(out_path)
    print(out_path)
    return

SOM_DIR="/home/ysorimachi/data/synfos/som/model/labels_synfos/train"
def select_kmeans_day(cele,kmeans_label=0):
    # k-meansで分類された16分類に該当する、64分類のsomクラスタに分類された日にちを取得して学習モデルとして実施する
    def list_som2kmeans():
        path = f"{SOM_DIR}/label_kmeans.csv"
        df = pd.read_csv(path)
        df= df[df[cele]==kmeans_label]
        if not df.empty:
            _cluster = df["cluster"].values.tolist()
        else:
            _cluster = []
        return _cluster
    
    def list_kmeans_dd(_cluster):
        path = f"{SOM_DIR}/label_som.csv"
        df = pd.read_csv(path)
        df["dd"] = df["dd"].apply(lambda x: str(x)[:8])
        df = df.loc[df[cele].isin(_cluster),:]
        if not df.empty:
            return df["dd"].values.tolist()
        else:
            return []
        
    _clstr64 = list_som2kmeans()
    _dd = list_kmeans_dd(_clstr64)
    return len(_dd),_dd

def select_som_day(lbl,cele="MSPP",n_dim=3,isCNN=1,cate="train"):
    # k-meansで分類された16分類に該当する、64分類のsomクラスタに分類された日にちを取得して学習モデルとして実施する
    """[summary]
    Args:
        lbl ([type]): [description]
        cele (str, optional): [description]. Defaults to "MSPP".
        n_dim (int, optional): [description]. Defaults to 3.
        isCNN (int, optional): [description]. Defaults to 1.
        cate (str, optional): [description]. Defaults to "train".

    Returns:
        [type]: [description]
    """
    def train_flg(df):
        #2021.11.19
        df["istrain"] = df["dd"].astype(str).apply(lambda x: 1 if pd.to_datetime(x).day %2==1 else 0)
        return df
    
    def list_som_dd():
        # path = f"{SOM_DIR}/label_som.csv"
        path = f"{SOM_DIR}/label_som_{n_dim}_{isCNN}.csv"
        df = pd.read_csv(path)
        df = train_flg(df)
        
        if cate=="train":
            df = df[df["istrain"]==1]
        elif cate=="test":
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

def rename_col(df,mtd=2):
    rename_hash = {
        f"mix{mtd}" :"MIX",
        f"syn" :"SYN",
        f"ecm{mtd}" :"EC",
        f"rCR0" :"CR0",
        f"obs" :"OBS",
    }
    df = df.rename(columns=rename_hash)
    return df

def weather_info(_dd):
    _df =[]
    for ini_j in _dd:
        df = load_weather_fcs(ecode=ecode,ini_j=ini_j,normalize=True)
        _df.append(df)
    df = pd.concat(_df,axis=0)
    return df

def interpolate30min(df,_cele):
    for c in _cele:
        df[c] = df[c].interpolate('linear')
    return df

#---------------------------------------------------------------------------------
FIT_DIR="/home/ysorimachi/data/synfos/som/model/fitting_synfos"
ESTIMATE1="/home/ysorimachi/data/synfos/som/model/estimate1"
MODEL1="/home/ysorimachi/data/synfos/som/model/m1"
MODEL2="/home/ysorimachi/data/synfos/som/model/m2"

def train_reg(cele="MSPP",ecode="ecmf001",resid2_name="lgb",n_dim=3,isCNN=1):
    """
    concat 
    2021.10.12 stat !!
    2021.11.19 update
    """
    FIT_DIR="/home/ysorimachi/data/synfos/som/model/fitting_synfos"
    
    def cut_time(df,st,ed):
        tmp = df[(df["hh_int"]>=st)&(df["hh_int"]<=ed)]
        return tmp
    
    def details_info(df):
        n_dd = df["day"].nunique()
        tmp = cut_time(df,900,1500)
        
        ci_mean = np.mean(tmp["OBS"]/tmp["CR0"])
        ci_std = np.std(tmp["OBS"]/tmp["CR0"])
        # ratio = df[df["hh_int"]==1200].groupby("mm").count()["time"]
        return n_dd,ci_mean,ci_std
    
    def clensing(df,subset_col):
        subset_col = x_col + [y_col]
        for c in subset_col:
            df[c] = df[c].apply(lambda x: isFloat(x))
        df = df.dropna(subset=subset_col)
        return df

    info_hash = {}
    list_ratio = []
    coef_hash = {}
    #-------------------------------------------------------------------
    df = load_rad(ecode)
    df = rename_col(df)
    N_ALL=int(np.ceil(df["day"].nunique()/2))
    # print(N_ALL)
    # sys.exit()

    for lbl in range(n_dim**2):
        # n,_dd = select_kmeans_day(cele=cele,kmeans_label=lbl) #64分類用
        n,_dd = select_som_day(lbl=lbl,cele=cele,n_dim=n_dim,isCNN=isCNN,cate="train") #64分類用]
        # print(lbl,n,_dd[:5])
        # sys.exit()
        # _,_dd_all = select_som_day(cele=cele,lbl=lbl,,istrain=False,istest=False) #64分類用]   
        # w1_all = weather_info(_dd_all)
        # data srt
        if n !=0 :
            wi = weather_info(_dd)
            # print(wi.columns[1:])
            # sys.exit()
            df_train = df.loc[df["day"].isin(_dd),:]
            # print(df_train.head())
            # sys.exit()
            # df_train = rename_col(df_train)
            # df_end = df_train.copy() #最終的に出力を行うもの
            nn_dd,ci_mean,ci_std = details_info(df_train)
            info_hash[lbl] = [nn_dd,ci_mean,ci_std]

            #model making-----------
            df_train = df_train.merge(wi,how="left",on="time")
            _cele = use_reg_weather_element() #利用するデータについて
            df_train = interpolate30min(df_train,_cele) #30分内挿の実施
            # print(df_train.head())
            # sys.exit()
            #---pred1(fit)---#
            x_col = ["SYN","EC","CR0"]
            y_col = "OBS"
            
            df_train = cut_time(df_train,st=700,ed=1700)
            df_train = clensing(df_train,subset_col = x_col+[y_col] ) #float&drop
            
            #---pred1(fit)---#
            X,y = df_train[x_col].values,df_train[y_col].values
            lr = LinearRegression(fit_intercept=False).fit(X,y)
            m1_path = f"{MODEL1}/lr_{ecode}_{cele}_cls{lbl}.pkl"
            save_model(m1_path,lr) #save
            # coef_hash[lbl] = lr.coef_
            df_train["PRED"] = lr.predict(X)
            df_train["resid"] = df_train[y_col] - df_train["PRED"]
            
            #---pred2(fit)---#
            df_train = clensing(df_train,subset_col=_cele + ["resid"])
            X2,y2 = df_train[_cele].values,df_train["resid"].values
            
            m2 = Resid2(name=resid2_name,pretrained=False)
            m2.fit(X2,y2)
            m2_path = f"{MODEL2}/{resid2_name}_{ecode}_{cele}_cls{lbl}.pkl"
            m2.save(m2_path)
            print(f"model:{cele}({n_dim})&{resid2_name} -> {lbl} is {n}(num) - {np.round(n*100/N_ALL,1)}[%]")
        else:
            print(f"model:{cele}({n_dim})&{resid2_name} -> {lbl} is {n}(num) - {np.round(n*100/N_ALL,1)}[%]")
            info_hash[lbl] = [0,9999,9999]
            

    #details --------
    res = pd.DataFrame(info_hash).T
    res.index.name = "cluster"
    res.columns = ["N_DD","CI_mean","CI_std"]
    res.to_csv(f"{FIT_DIR}/info_{ecode}_{cele}.csv")
    return



def predict_rad(cele="MSPP",ecode="ecmf001",resid2_name="lgb",n_dim=3,isCNN=1):
    """
    concat 
    2021.10.12 stat !!
    """
    # FIT_DIR="/home/ysorimachi/data/synfos/som/model/fitting_synfos"
    def predict_calc(df,m1,m2):
        x_col = ["SYN","EC","CR0"]
        # x2_col = ["70RH","85RH","70UU","70VV","LOCA","MICA"]
        # _cele = ["70RH","85RH","70UU","70VV","LOCA","MICA"] #2021.10.15
        x2_col = use_reg_weather_element()
        # y_col = "OBS"
        X = df[x_col]
        X2 = df[x2_col]
        
        df["PRED1"] = df["MIX"] #初期値
        df["RESID1"] = 0.
        df["PRED2"] = df["MIX"] #初期値
        
        
        def clensing1(x):
            if x["PRED1"]<0:
                return 0
            elif x["PRED1"] > x["CR0"]:
                return x["CR0"]
            else:
                return x["PRED1"]
            
        def clensing2(x):
            if x["PRED1"]==0:
                return 0
            elif np.abs(x["RESID1"]) > 0.2*x["CR0"]:
                return x["PRED1"]
            elif x["PRED1"] + x["RESID1"] > x["CR0"]:
                return x["PRED1"]
            else:
                return x["PRED1"] + x["RESID1"]
        # print(X.head())
        
        # _r=[]
        # for i,r in  df.iterrows():
        #     if r["SYN"] != 9999. and r["EC"] != 9999. and r["CR0"] != 9999.:
        #         _r.append(r["MIX"])
        #     else:
        df["PRED1"] = m1.predict(X)
        df["PRED1"] = df.apply(clensing1,axis=1)
        df["RESID1"] = m2.predict(X2)
        df["PRED2"] = df.apply(clensing2,axis=1)
        return df

    def cut_time(df,st,ed):
        tmp = df[(df["hh_int"]>=st)&(df["hh_int"]<=ed)]
        return tmp
    
    def details_info(df):
        n_dd = df["day"].nunique()
        tmp = cut_time(df,900,1500)
        
        ci_mean = np.mean(tmp["OBS"]/tmp["CR0"])
        ci_std = np.std(tmp["OBS"]/tmp["CR0"])
        # ratio = df[df["hh_int"]==1200].groupby("mm").count()["time"]
        return n_dd,ci_mean,ci_std
    
    def clensing(df,subset_col):
        # subset_col = x_col + [y_col]
        for c in subset_col:
            df[c] = df[c].apply(lambda x: isFloat(x))
        df = df.dropna(subset=subset_col)
        return df

    info_hash = {}
    list_ratio = []
    coef_hash = {}
    
    _df = []
    rad = load_rad(ecode)
    rad = rename_col(rad)
    for lbl in tqdm(list(range(n_dim**2))):
        # n,_dd = select_kmeans_day(cele=cele,kmeans_label=lbl) #64分類用
        # n0,_dd0 = select_som_day(cele=cele,lbl=lbl,istrain=True) #train
        # n1,_dd1 = select_som_day(cele=cele,lbl=lbl,istrain=False,istest=True) #test  
        # n,_dd = select_som_day(cele=cele,lbl=lbl,istrain=False,istest=False) #test
        n,_dd = select_som_day(lbl=lbl,cele=cele,n_dim=n_dim,isCNN=isCNN,cate="all") #64分類用]
        # data srt
        if n:
            # print("CLUSTER",lbl ,f"is [ {n} ] ",cele,ecode,resid2_name)
            wi = weather_info(_dd)
            df = rad.loc[rad["day"].isin(_dd),:]
            # print(df)
            # sys.exit()
            m1 = load_model(f"{MODEL1}/lr_{ecode}_{cele}_cls{lbl}.pkl")
            m2 = Resid2(name=resid2_name,pretrained=f"{MODEL2}/{resid2_name}_{ecode}_{cele}_cls{lbl}.pkl")
            #model making-----------
            df = df.merge(wi,how="left",on="time")
            x_col = ["SYN","EC","CR0"]
            _cele = use_reg_weather_element() #2021.11.19
            df = interpolate30min(df,_cele) #30分内挿の実施
            df = cut_time(df,600,1800)
            df = df.replace(9999,np.nan)
            # print(df.shape)
            df = clensing(df,subset_col = x_col +_cele)
            # print(df.shape)
            df = df[["time","OBS","MIX"]+x_col+_cele]
            # print(df.isnull().sum())
            # print(df.head())
            # sys.exit()
            #---pred1(fit)---#
            
            df = predict_calc(df,m1,m2)
            df["CLUSTER"] = lbl
            # df = train_flg(df)
            _df.append(df)

        else:
            # print("CLUSTER",lbl ,"is [ 0 ] ",cele,ecode,resid2_name)
            pass
        
    df = pd.concat(_df,axis=0)
    df = df.sort_values("time")
    df.to_csv(f"{ESTIMATE1}/rad_{resid2_name}_{ecode}_{cele}_{n_dim}_{isCNN}.csv", index=False)
    # print(datetime.now(),"[end]")
    return
    
# _img,_time = mk_DataLoader("sp",_
def model_cleaner():
  subprocess.run(f"rm {MODEL1}/*.pkl",shell=True)
  subprocess.run(f"rm {MODEL2}/.pkl",shell=True)
  return

# _img,_time = mk_DataLoader("sp",_
def rad_cleaner():
  subprocess.run(f"rm {ESTIMATE1}/*.csv",shell=True)
  subprocess.run(f"rm ./*.out",shell=True)
#   subprocess.run(f"rm {MODEL2}/.pkl",shell=True)
  return

def use_reg_weather_element():
    """[summary]
    回帰に利用する説明変数を指定する部分
    Returns:
        [type]: [description]
    """
    _cele = ["70RH","85RH","70UU","70VV","LOCA","MICA"] #2021.10.15
    _cele = ['30RH', '50RH', '70OO', '70RH', '70UU', '70VV', '85OO', '85RH', '85UU','85VV', 'HICA', 'LOCA', 'MICA', 'MSPP']
    # _cele = ["70RH","85RH","70UU","70VV","LOCA","MICA"]
    return _cele

if __name__== "__main__":
    
    if 0:
        size,FD,ndim=56,39,5 #2021.10.12
        # FD=39: 39 時間後翌日正午
        # FD=39: 15 時間後当日正午
        # Fitting ------------
        for n_dim in [3,5]:
            predict_som_cluster(size=size,FD=FD,ndim=n_dim,isCNN=1) #出力していくイメージ
        sys.exit()
    
    if 0:
        n_clusters=9
        concat_cluster(n_clusters=n_clusters)
        # sys.exit()
    
    if 1:
        # ecode="ecmf001"
        # model_cleaner()
        #----------------------------------------
        log_path = "./log_a01_03.log"
        log_write(log_path,"start! ",init=True)
        # _cele = ["MSPP","LOCA","MICA","HICA"]
        
        _ecode,_scode,_name= load_10()
        _ecode = ["ecmf003"] #東京のみ
        _cele = ["MSPP","HICA"]
        _resid2 = ["lasso","ridge","svr","tree","rf","lgb","mlp3","mlp5"]
        # _resid2 = ["mlp3","mlp5"]
        # -----------------------------------------
        # Debug --- ----------
        # _cele = ["MSPP"]
        # _ecode = _ecode[:1]
        #debug ---
        n_dim=5
        for resid2_name in _resid2:
            for ecode in _ecode:
                for cele in _cele:
                    #Fitting------------------
                    train_reg(cele=cele,ecode=ecode,resid2_name=resid2_name,n_dim=n_dim,isCNN=1)
                    # estimate_rad(cele=cele,ecode=ecode) #old (Dont' use !)
                    #predict------------------
                    predict_rad(cele=cele,ecode=ecode,resid2_name=resid2_name,n_dim=n_dim,isCNN=1)
                    print(datetime.now(),"[END]",ecode,cele)
                    log_write(log_path,f"[END] {ecode} {cele} name={resid2_name}",init=False)
                    # sys.exit()
                # sys.exit()
            
        