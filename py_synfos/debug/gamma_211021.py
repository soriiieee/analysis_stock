# -*- coding: utf-8 -*-
# when   : 2020.03.23 
# who : [sori-machi]
# what : [ ]
"""

gamma　functionの学習用のサイト
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html
・東京大学統計学配布資料->PDF[https://elf-c.he.u-tokyo.ac.jp/courses/244/files/6852?module_item_id=3390]
・東京大学統計学学習配布サイト->動画リスト[http://www.mi.u-tokyo.ac.jp/consortium/e-learning.html]

"""
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
from convSokuhouData import conv_sfc #(df, ave=minutes,hour)
# #amedas relate 2020.02,04 making...
from tool_AMeDaS import code2name, name2code, name2scode
# from tool_110570 import get_110570,open_110570
# from tool_100571 import get_100571,open_100571
#(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
import subprocess
# outd ='/home/griduser/work/sori-py2/deep/out/0924_01'
# os.makedirs(outd, exist_ok=True)
# subprocess.run('rm -f *.png', cwd=outd, shell=True)

#scipy module import 
import scipy.stats as ss 



def isFloat(x):
  try:
    return float(x)
  except:
    return np.nan

def clensing(df):
  df = df.replace(9999,np.nan)
  for c in df.columns:
    df[c].apply(lambda x: isFloat(x))
  return df

def load_data(month="202102",name="東京"):
  #-------load sfc Dataframe ---
  scode = name2scode(name)
  sfc_path = f"/work/ysorimachi/make_SOKUHOU3/out/{month}/sfc2/sfc_10minh_{month}_{scode}.csv"
  if os.path.exists(sfc_path):
    df = pd.read_csv(sfc_path)
    df = conv_sfc(df)
  else:
    df = pd.DataFrame()
  #-------個別案件---
  use_col = ["time",'windDirection', 'windSpeed', 'temp']
  if df.shape[0] !=0:
    df = df[use_col]
    df = df.set_index("time")
    df = clensing(df) #data clensing ---
  
  for c in df.columns:
    df[c] = df[c].apply(lambda x: np.nan if np.abs(x) > 100 else x)
  #-------個別案件---
  # names=["time","rrad","isyn","iecm","iecc","flg_ecm","flg_ecc"]
  return df

def plot_line(ax,df,title="plot"):
  # for col in ["rrad","isyn","iecm"]:
  for col in ["rrad","isyn","iecm","rCR0","rS0"]:
    if col =="rrad":
      ax.plot(df[col].values, label=col, lw=5)
    else:   
      ax.plot(df[col].values, label=col)
  ax.set_ylim(0,1000)
  ax.set_ylabel("rad[W/m2]")
  ax.set_xlabel("Forecast Time[jt]")
  ax.set_title(title)
  ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
  return ax


def plot_module(df,png_path="./sample.png"):
  f,ax = plt.subplots(int(len(_dd)/2),2,figsize=(8,8))
  ax = ax.flatten()
  for j,dd in enumerate(_dd):
    path = f"{DIR}/{dd}/{cpnt}.dat"
    df = load_data(path)
    
    title = f"[{cpnt}]({dd})JST"
    ax[j] = plot_line(ax[j],df,title=title)
      # print(df.head())
  # plt.subplots_adjust(wspace=0.4, hspace=0.5)
  plt.subplots_adjust(hspace=0.8,wspace=0.7)
  plt.savefig(png_path,bbox_inches="tight")
  print("[END] -> ",png_path)
  return 

def plot_hist(df,_col,png_path="./sample.png"):
  N=len(_col)
  
  if N>1:
    #---multi plot ---
    f,ax = plt.subplots(N,1,figsize=(15,N*5))
    for i,c in enumerate(list(_col)):
      sns.distplot(df[c],ax=ax[i],label=c)
      ax[i].legend()
  else:
    #---mono plot ---
    f,ax = plt.subplots(figsize=(15,10))
    sns.distplot(df[_col[0]],ax=ax,label=_col[0])
    ax.legend()
  f.savefig(png_path,bbox_inches="tight")
  return 
  


def test():
  
  #-------------γ分布製造機---------------
  def gamma_generator(a,b=1):
    """
    a: shape factor
    b: scale factor
    """
    # mean,var,skew,kurt = ss.gamma.stats(a, moments ='mvsk')
    # print(mean,var,skew,kurt)
    X0,X1 = ss.gamma.ppf(0.01,a,scale=b), ss.gamma.ppf(0.99,a,scale=b)
    X = np.linspace(X0,X1,100)
    p = ss.gamma.pdf(X,a,scale=b)
    return X,p
  
  def gamma_generator2(a,b=1):
    seed=1
    p = ss.gamma.rvs(a,size=1000,scale=b,random_state=seed)
    return p
  #---------------
  
  f,ax = plt.subplots(figsize=(18,8))
  for a in [1,2]:
    for b in [1,2,3]:
      # X,p = gamma_generator(a,b)
      p = gamma_generator2(a,b)
      
      ax.plot(p,label=f"a={a} | b={b}")
  ax.legend(loc="upper right")
  f.savefig("./sample.png", bbox_inches="tight")
  # print("0.01 -> ",ss.gamma.ppf(0.01,a))
  # print("0.99 -> ",ss.gamma.ppf(0.99,a))
  # sys.exit()
  return

def test2():
  seed = 1
  #-------------γ分布製造機---------------
  def gamma_generator2(a,b=1, size=1000):
    data = ss.gamma.rvs(a,size=size,scale=b, random_state = seed)
    return data 
  #---------------
  
  _a = [1,2,3]
  _b = [1,2,3]
  
  f,ax = plt.subplots(len(_a),len(_b),figsize=(18,8))
  for i,a in enumerate(_a):
    for j,b in enumerate(_b):
      p_data = gamma_generator2(a,b,size=100)
      params = ss.gamma.fit(p_data)
      
      a_hat = np.round(params[0],3)
      loc_hat= params[1]
      b_hat= np.round(params[2],3)
      
      fit_gamma = ss.gamma.freeze(*params)
      
      X = np.linspace(0,30,1000)
      
      ax[i,j].hist(p_data, bins=30,alpha=0.7,density=True,label=f"data-a={a}/b={b}")
      ax[i,j].plot(X, fit_gamma.pdf(X), label=f"fitted-a={a_hat}/b={b_hat}")
      ax[i,j].legend(loc="upper right")
  f.savefig("./sample.png", bbox_inches="tight")
  # print("0.01 -> ",ss.gamma.ppf(0.01,a))
  # print("0.99 -> ",ss.gamma.ppf(0.99,a))
  # sys.exit()
  return

def main():
  """
  # ---------------------------------------------
  # main function ------------------ 
  init 21.03.23
  update 21.06.26
  
  # ---------------------------------------------
  # fitting data --
  # [ fitter -> 実データを確率分布でfitさせるのが目的 ] 
  https://ichi.pro/python-no-fitter-raiburari-o-shiyoshite-de-ta-ni-saitekina-bunpu-o-mitsukeru-54539005293289
  pip install fitter
  pip install damona
  """
  def ammong_mean(X):
    X = np.array(X)
    X2 = [ (X[i-1]+X[i])/2 for i in range(1,X.shape[0])]
    X2 = np.array(X2)
    return X2
  
  from fitter import Fitter
  
  _df = [ load_data(month=str(mm)) for mm in range(202102,202102+3+1) ]
  df = pd.concat(_df,axis=0)
  
  df = df.dropna()
  X = df["windSpeed"].values
  p,X_range = np.histogram(X, bins=30,density=True)
  X_range = ammong_mean(X_range) # -> 隣接間 の 平均ファイル
  
  # print(p.shape,X_range.shape,X2.shape)
  if 0: #debug ---
    plt.plot(X_range,p,marker="o")
    plt.savefig("./sample.png", bbox_inches="tight")
  
  # fitting ----
  distributions = ['gamma', 'rayleigh', 'uniform','beta']
  fm = Fitter(p,distributions=distributions)
  fm.fit() #fitting several model
  params = fm.fitted_param['gamma']
  a_hat,_,b_hat = fm.fitted_param['gamma']

  for dist in distributions:
    params = fm.fitted_param[dist]
    print(dist,"--> ",params)
  
  
  # plotting ----
  f,ax = plt.subplots(figsize=(18,8))
  
  _x = np.linspace(np.min(X),np.max(X),30)
  # # print(np.min(X),np.max(X))
  # print(_x)
  # sys.exit()
  # sys.exit()
  
  ax.hist(X, bins=30,alpha=0.7,density=True,label=f"WIND SPEED")
  # ax.plot(_x,p,label=f"WIND SPEED")
  for dist in distributions[:1]:
    dist="gamma"
    
    
    # print(p)
    
    # sys.exit()
    params = ss.gamma.fit(p)
    a,loc,b = params
    dist_model = ss.gamma.freeze(a,scale=b)
    _p = ss.gamma.rvs(a,loc=loc,scale=b,size=1000)
    # print(params)
    # sys.exit()
    # ax.plot(dist_model.pdf(_x), label=f"{dist}")
    ax.hist(_p, label=f"{dist}")
    # p_values = fm.fitted_pdf['gamma']
    # p_values = fm.fitted_pdf[dist]
  # print(np.min(p_values), np.max(p_values), p_values.shape)
  # sys.exit()
  ax.legend(loc="upper right")
  f.savefig("./sample.png", bbox_inches="tight")
  print("end")
  return




if __name__ == "__main__":
  main()
  # test2()
  