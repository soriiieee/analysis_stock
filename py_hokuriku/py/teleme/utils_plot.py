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
from convSokuhouData import conv_sfc #(df, ave=minutes,hour)
#amedas relate 2020.02,04 making...
from tool_AMeDaS import code2name, name2code
from tool_110570 import get_110570,open_110570
from tool_100571 import get_100571,open_100571
#(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
import subprocess

def plot_pu_line():
  """ 
  2021.09.02 
  2021.09.08 with- smame も実装済 
  """
  f,ax = plt.subplots(3,4,figsize=(4*5,3*4))
  ax = ax.flatten()
  _mm19 = loop_month(st="201904")[:12]
  rcParams['font.size'] = 14
  
  for i,mm in enumerate(_mm19):
    
    # line
    X_line = np.linspace(0,1,1000)
    _label = ["19年05月(unyo)","月別(unyo)","月別(8now0)","月別(合算)(8now0)"]
    
    for j ,lbl in enumerate(_label):
      
      if j==0:
        month, cate= "201905","obs"
        color,alpha,ls= "gray",0.8,"--"
      if j==1:
        month, cate= mm,"obs"
        color,alpha,ls= "green",0.8,"-"
      if j==2:
        month, cate= mm,"8now0"
        color,alpha,ls= "blue",1,"-"
      if j==3:
        month, cate= mm,"8now0_with_smame"
        color,alpha,ls= "red",1,"-"
        
      a,b,c = get_a_b_c(month,cate=cate)
      
      pu_pred0 = a*X_line**2 + b*X_line + c
      ax[i].plot(X_line,pu_pred0,lw=1,color=color, label=lbl, alpha=alpha, linestyle = ls)
      # pu_pred1 = a1*X_line**2 + b1*X_line + c1
      # pu_pred2 = lr.predict(pf.fit_transform(X_line.reshape(-1,1)))
    # if 1:
    #   ax[i].plot(X_line,pu_pred0,lw=1,color="k", label="201905")
    #   ax[i].plot(X_line,pu_pred1,lw=1,color="b",label="per-Month")
    #   ax[i].plot(X_line,pu_pred2,lw=1,color="r", label="2020-ver")
    
    ax[i].legend(loc="lower right",fontsize=8)
    ax[i].set_title(f"{mm}(p.u reg)")
    
    ax[i].set_ylim(0,1)
    ax[i].set_xlim(0,1)
    ax[i].set_ylabel("p.u.")
    ax[i].set_xlabel("平均日射量[kW/m2]")
    print(datetime.now(),"[end]", mm)
    # sys.exit()
    #---------------------------------------------------------
  plt.subplots_adjust(wspace=0.4, hspace=0.4)
  f.savefig(f"{OUTDIR}/line_pu_2rad_all2.png",bbox_inches="tight")
  print(OUTDIR)
  

def plot_pu_line2(m_shift=1, h_shift=1):
  _hh = list_hh(st="0530",ed="1830")[1:-1]
  _mm = loop_month()[:12]
  n_mm,n_hh= len(_mm), len(_hh)
  
  #params ---------
  path = f"{OUT_HOME}/param/regPara_m{m_shift}_h{h_shift}.csv"
  if not os.path.exists(path):
    reg_pu_per_hh(m_shift=m_shift, h_shift=h_shift)
  else:
    print("already making")
  par = pd.read_csv(path)
  # print(par.shape, "_hh=", len(_hh), "_mm=", len(_mm))
  # sys.exit()
  
  # par["mm"] =par["mm"].astype(str)
  # par["hh"] =par["hh"].apply(lambda x: str(x).zfill(4))
  #params ---------
  def load_abc(mm,hh):
    # tmp = par[(par["mm"]==str(mm))&(par["hh"])]
    tmp = par[(par["mm"]==int(mm))&(par["hh"]==int(hh))]
    a = tmp["a"].values[0]
    b = tmp["b"].values[0]
    c = tmp["c"].values[0]
    return a,b,c
  
  # f,ax = plt.subplots(n_hh,n_mm, figsize=(n_mm*2.1,n_hh*1.6))
  f,ax = plt.subplots(figsize=(12,12))
  # debug ---
  # _hh,_mm=_hh[:2],_mm[:2]
  
  for i, hh in enumerate(_hh):
    for j,mm in enumerate(_mm):
      
      # ax2 = ax[i,j]
      # df = pd.read_csv(f"{OUT_HOME}/csv/df/set_mm{mm}_hh{hh}.csv")
      a,b,c = load_abc(mm,hh)
      # ax.scatter(df["obs"], df["p.u"], s=1,color="b")
      _x = np.linspace(0,1,1000)
      _y = a*_x**2 + b*_x + c
      # ax.plot(_x,_x, color="k", lw=1)
      ax.plot(_x,_y, color="gray",alpha=0.5, lw=1)
      ax.set_title(f"PU_regression(degree=2) 12 month * 25 time = 300")
      ax.set_ylim(0,1)     
      ax.set_xlim(0,1)
      ax.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
      print(datetime.now(), "-"*10,hh,mm)
  
  ax.plot(_x,_x, color="k",alpha=1, lw=5)
  ax.set_xlabel(r"OBS[kW/$m^2$]")
  ax.set_ylabel(r"P.U[-]")
  
  # plt.subplots_adjust(wspace=0.4, hspace=0.4)
  f.savefig(f"{OUT_HOME}/png/reg_pu_line_m{m_shift}_h{h_shift}.png", bbox_inches="tight")
  print(f"{OUT_HOME}/png/reg_pu_line_m{m_shift}_h{h_shift}.png")
  return