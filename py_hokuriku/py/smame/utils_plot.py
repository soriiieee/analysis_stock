# -*- coding: utf-8 -*-
# when   : 2021.07.15
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
#amedas relate 2020.02.04 making...
from tool_AMeDaS import code2name, name2code
# from tool_110570 import get_110570,open_110570
# from tool_100571 import get_100571,open_100571
#(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
import subprocess
# outd ='/home/griduser/work/sori-py2/deep/out/0924_01'
# os.makedirs(outd, exist_ok=True)
# subprocess.run('rm -f *.png', cwd=outd, shell=True)
sys.path.append('/home/ysorimachi/work/hokuriku/py')
from utils import *
from mapbox import map_lonlat3 #(df,html_path,zoom=4,size_max=None)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

from matplotlib import rcParams
rcParams['font.size'] = 15
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

SMAME_SET3="/work/ysorimachi/hokuriku/dat2/smame/set3_ts" # 30min_201912.csv
SMAME_DD="/home/ysorimachi/work/hokuriku/py/smame"

def plot_pu_line():
  """ 2021.09.02 """
  f,ax = plt.subplots(3,4,figsize=(4*5,3*4))
  ax = ax.flatten()
  _mm19 = loop_month(st="201904")[:12]
  rcParams['font.size'] = 14
  
  for i,mm in enumerate(_mm19):
    
    # line
    X_line = np.linspace(0,1,1000)
    _label = ["19年05月(unyo)","19年05月(8now0)","月別(unyo)","月別(8now0)"]
    
    #scatter
    df = get_smame_rad(cate="surplus",mm=mm,radname="8now0")
    df = df.dropna()
    ax[i].scatter(df["obs"],df["p.u"],s=2,color="k",alpha=0.8)
    # print(df.describe())
    # sys.exit()
    
    
    for j ,lbl in enumerate(_label):
      
      if j==0:
        month, cate= "201905","obs"
        color,alpha,ls= "gray",0.8,"--"
      if j==1:
        month, cate= "201905","8now0"
        color,alpha,ls= "gray",0.8,"-."
      if j==2:
        month, cate= mm,"obs"
        color,alpha,ls= "blue",1,"-"
      if j==3:
        month, cate= mm,"8now0"
        color,alpha,ls= "red",1,"-"
        
      a,b,c,d = get_a_b_c_d(month,cate=cate)
      
      # pu_pred0 = a*X_line**2 + b*X_line + c
      pu_pred0 = a*X_line**3 + b*X_line**2 + c*X_line + d
      ax[i].plot(X_line,pu_pred0,lw=1,color=color, label=lbl, alpha=alpha, linestyle = ls)
      # pu_pred1 = a1*X_line**2 + b1*X_line + c1
      # pu_pred2 = lr.predict(pf.fit_transform(X_line.reshape(-1,1)))
    # if 1:
    #   ax[i].plot(X_line,pu_pred0,lw=1,color="k", label="201905")
    #   ax[i].plot(X_line,pu_pred1,lw=1,color="b",label="per-Month")
    #   ax[i].plot(X_line,pu_pred2,lw=1,color="r", label="2020-ver")
    
    # ax[i].legend(loc="lower right",fontsize=8)
    ax[i].legend(loc="upper left",fontsize=8)
    ax[i].set_title(f"{mm}[smame(余剰)]")
    
    ax[i].set_ylim(0,1)
    ax[i].set_xlim(0,1)
    ax[i].set_ylabel("p.u.")
    ax[i].set_xlabel("平均日射量[kW/m2]")
    print(datetime.now(),"[end]", mm)
    # sys.exit()
    #---------------------------------------------------------
  plt.subplots_adjust(wspace=0.5, hspace=0.5)
  f.savefig(f"{OUTDIR}/line_pu_2rad_all.png",bbox_inches="tight")
  print(OUTDIR)




if __name__ == "__main__":
  
  if 0:
    # mk_reg_para(cate="surplus")
    plot_pu_line()
    
  if 1:
    for cate in ["all", "surplus"]:
    # for cate in ["surplus"]:
    # cate = "surplus"
      estimate(cate=cate,radname="8now0")