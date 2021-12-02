import numpy as np 
import pandas as pd 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gc

import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')
# matplot で日本語
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 18
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']


def load_df(path):
  df = pd.read_csv(path)
  if df["time"].dtypes==object:
    df["time"] = pd.to_datetime(df["time"])
#     sus.exit()
  df = df.set_index("time")
  return df

def plot(df,_col, title=None):
    f,ax = plt.subplots(figsize=(20,8))
    for c in _col:
        ax.plot(df[c], label=c)
    
    if title:
        ax.set_title(title)
    ax.legend()
    plt.show()
    return

def plot2(_df,_col,_title=None,ylim=None):
    f,ax = plt.subplots(figsize=(20,16))
    for i,df in enumerate(_df):
        for c in _col:
            if _title:
                label=f"{_title[i]}-{c}"
            else:
                label=c
            ax.plot(df[c], label=label)
    
#     if title:
#         ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend()
    plt.show()
    return



def cut_time(df,st,ed=None):
    if not "time" in df.columns:
        df= df.reset_index()
    if df["time"].dtypes==object:
        df["time"] = pd.to_datetime(df["time"])
    
    if ed ==None:
        ed = datetime.now()
    df = df[(df["time"]>st)&(df["time"]<ed)]
    if "time" in df.columns:
        df = df.set_index("time")
    return df

def change_price(df):
    ini_p = df["close"].values[0]
    df["close"] /=ini_p
    return df

from scipy.stats import norm
import seaborn as sns
def main(size=100):
    x = norm.rvs(loc=0,scale=1,size=size)
    if type(x) != np.ndarray:
        x = np.array(x) 
    return x

def plot_scatter1(x, title="サンプル"):
    plt.figure(figsize=(18,8))
    plt.scatter(np.arange(len(x)),x)
    plt.title(title)
    plt.show()
    return


def plot_bins(x,title="sample"):
    f,ax = plt.subplots(figsize=(18,8))
    sns.histplot(x,ax=ax)
    plt.show()
    return

import statsmodels.api as sm
# from statsmodels.graphics.gofplots import qqplot
from scipy.stats import norm,gamma
# x = main(size=50)


def get_stock(keyword,st,ed=None):
    path = [ f for f in _path if keyword in f]
    if len(path)==0:
        sys.exit("No FILE !")
    else:
        path = path[0]
        pass
    df = load_df(path)
    df = cut_time(df,st=st, ed=ed)
    return df

import pykakasi
def convert_roma(word):
    kks = pykakasi.kakasi()
    result = kks.convert(word)
    return result[0]['hepburn']

def plot_multi(_df,c,_title=None):
    f,ax = plt.subplots(figsize=(20,15))
    for j,df in enumerate(_df):
        if _title:
            label=_title[j]
        else:
            label="sample"
        ax.plot(df[c], label=label)
    
#     ax.legend()
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.title(c)
    plt.show()
    return

def plot_vola(_df,c,_title=None, isPrint=False):
    f,ax = plt.subplots(figsize=(20,8))
    
    _df2=[]
    for j,(df,title) in enumerate(zip(_df,_title)):
        df["com"] = title
        _df2.append(df)
        
        if isPrint:  
            mu = np.round(np.mean(df[c]),3)
            std = np.round(np.std(df[c]),3)
            print(title,f"-> mu = {mu} | std = {std}")
    
    df = pd.concat(_df2, axis=0)
    
    ax = sns.boxplot(x="com", y=c, data=df,ax=ax)
#     ax.legend()
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    plt.show()
    return

def plot_corr(_df,c,_title=None):
    _df2=[]
    f,ax = plt.subplots(figsize=(20,20))
    for j,(df,title) in enumerate(zip(_df,_title)):
#         s = df[c]
        df.index.name = "time"
        lbl = f"{j}_{title}"
        df =df.rename(columns={c: lbl})
        s = df[lbl]
        _df2.append(s)
    
    df = pd.concat(_df2,axis=1)
    corr = df.corr()
    if 0:
        _index = corr.index
        _columns = corr.columns
        corr = np.where(corr.values>0.7,1,0)
        corr = pd.DataFrame(corr,index=_index,columns=_columns)
    sns.heatmap(corr,annot=True,ax=ax, cmap="seismic",vmin=-1,vmax=1)
#     ax = sns.pairplot(df,ax=ax)
    plt.show()
    return

from scipy.stats import shapiro

def plot_shapiro(_df,c,_title,isPrint=False):
    f,ax = plt.subplots(figsize=(20,8))
    
    _df2=[]
    _w,_p=[],[]
    for j,(df,title) in enumerate(zip(_df,_title)):
        
        W,p = shapiro(df[c])
        _w.append(W)
        _p.append(p)
        
        if isPrint:  
            mu = np.round(W,3)
            std = np.round(p,3)
            print(title,f"-> W = {W} | p-values = {p}")
            
#     ax.legend()
    ax.bar(np.arange(len(_df)), _p)
    ax.set_xlim(0,len(_df))
    ax.set_xticks(np.arange(len(_df)))
    ax.set_xticklabels(_title, rotation=90)
    ax.set_ylim(0,0.3)
    ax.axhline(y=0.10,color="r", lw=1) #優位水準　10以上で受容/正規分布を有しているとみなせる
    
#     ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
#     ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    plt.show()
    return

#自己相関の確認　->Ljung-box の検定
from statsmodels.stats.diagnostic import acorr_ljungbox

def plot_LjungBox(_df,c,_title,isPrint=False):
    #　自己相関の有無の検定
#     帰無仮説[自己相関なし]　--
#     p-value　大：　自己相関なし
#     p-values　小：　自己相関の可能性
    f,ax = plt.subplots(figsize=(20,8))
    
    _df2=[]
    _w,_p=[],[]
    for j,(df,title) in enumerate(zip(_df,_title)):
        
        lbvalues, pvalues = acorr_ljungbox(df[c].values, lags=50)
#         _w.append(W)
#         _p.append(p)
        
        if isPrint:  
            p= pvalues[0]
            print(title,f"-> Ljung-Box | p-values = {p}")
            
#     ax.legend()
    ax.bar(np.arange(len(_df)), _p)
    ax.set_xlim(0,len(_df))
    ax.set_xticks(np.arange(len(_df)))
    ax.set_xticklabels(_title, rotation=90)
    ax.set_ylim(0,0.3)
    ax.axhline(y=0.10,color="r", lw=1) #優位水準　10以上で受容/正規分布を有しているとみなせる
    
#     ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
#     ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    plt.show()
    return

def plot_2com_2data(_key,st,ed):
    # 収益率の時間依存を確認する為のscatter
    _df,_title=[],[]
    lag1,lag2=1,5
    n_com = len(_key)
    f,ax=plt.subplots(2,2,figsize=(18,18))
    for i,key in enumerate(_key):
        title=convert_roma(key)
#         kwyword = os.path.basename(p).split(".")[0].split("_")[1]
        df = get_stock(key,st=st,ed=ed)
        ax[0,i].scatter(df["log_diff"],df["log_diff"].shift(lag1),color="r")
        ax[0,i].plot(df["log_diff"],df["log_diff"],color="k", lw=1)
        ax[0,i].set_title(f"{title}({lag1})")
        
        ax[1,i].scatter(df["log_diff"],df["log_diff"].shift(lag2),color="r")
        ax[1,i].plot(df["log_diff"],df["log_diff"],color="k", lw=1)
        ax[1,i].set_title(f"{title}({lag2})")
    
    ax=ax.T
    plt.show()
    return

