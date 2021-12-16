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

sys.path.append("..")
from som_data.a01_99_utils import *
# from som_data.c01_som_cluster import *
from som_data.x99_pre_dataset import load_rad, load_10
from Err_rad_csv import load_predict_Rad
from utils import *


TMP="/home/ysorimachi/data/synfos/tmp/ci/details"

def main(n_cate):
  _dd,_ll = load_cate(n_cate,train="ALL")
  
  
  vcol = "OBS"
  df = pd.DataFrame()
  df["dd"] = _dd
  df["ll"] = _ll
  
  
  _lbls = sorted(df["ll"].unique().tolist())
  rad = load_predict_Rad("ecmf003",n_cate,"cloud3")
  rad["hhmm"] = rad["time"].apply(lambda x: x.strftime("%H:%M"))
  rad["DIFF"] = rad["MIX"] - rad["OBS"]
  rad["CI"] = rad["OBS"] / rad["CR0"]
  
  if n_cate ==1:
    # cut_bins =[-1000,-200,-100,-50,0,50,100,200,1000]
    cut_bins =[-1000,-500,-300,-150,0,150,300,500,1000]
    lbl_out = [ f"{cut_bins[i]}~{cut_bins[i+1]}"for  i in range(len(_lbls))]
    lbl_out = [f"CLS{i}({lbl_out[i]})" for i in range(len(_lbls))]
    
    tmp = df["ll"].value_counts().reset_index()
    tmp.columns = ["CLUSTER","count"]
    tmp = tmp.sort_values("CLUSTER")
    tmp["name"] = lbl_out
    tmp.to_csv(f"{TMP}/bins_cate{n_cate}.csv")
    vmin,vmax = -500,500
    title = "OULIER BINS [ MAX-DIFF ] (per hour)"
    xlabel = r"DIFF (MIX-OBS) [W/$m^2$]"
  
  if n_cate==2:
    tmp = df["ll"].value_counts().reset_index()
    tmp.columns = ["CLUSTER","count"]
    tmp = tmp.sort_values("CLUSTER")
    
    cut_bins =[-1,0.2,0.4,0.6,0.9,1.5]
    lbl_out = [ f"{cut_bins[i]}~{cut_bins[i+1]}"for  i in range(len(_lbls))]
    lbl_out = [f"CLS{i}({lbl_out[i]})" for i in range(len(_lbls))]
    tmp["name"] = lbl_out
    tmp.to_csv(f"{TMP}/bins_cate{n_cate}.csv")
    vmin,vmax = 0,1200
    title = "Solar Radiation [ CI actegory ] (per hour)"
    xlabel = r"Solar Radiation [W/$m^2$]"

  _df =[]
  for ll in _lbls:
    _dd = df[df["ll"]==ll]["dd"].values.tolist()
    
    tmp = rad.loc[rad["dd"].isin(_dd),:]
    if n_cate == 1:
      tmp = tmp.groupby("hhmm").agg({"DIFF": "mean"})
    else:
      tmp = tmp.groupby("hhmm").agg({vcol: "mean"})
    _df.append(tmp)

  df = pd.concat(_df,axis=1)
  df.columns = lbl_out
  
  f,ax = plt.subplots(figsize=(15,8))
  for c in df.columns:
    ax.plot(df[c],marker="o", label=c)
  ax.set_ylim(vmin,vmax)
  ax.set_xticks(np.arange(0,len(df)))
  ax.set_xlim(0,len(df))
  ax.set_xticklabels(df.index, fontsize=12, rotation=45)
  ax.set_title(title)
  ax.set_xlabel(xlabel)
  plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
  f.savefig(f"{TMP}/ave_cate{n_cate}.png", bbox_inches="tight")
  

def resid_rad(dd="20190620"):
  
  df = load_predict_Rad("ecmf003",1,"cloud3")
  df["DIFF"] = df["MIX"] - df["OBS"]
  df = df[df["dd"] == dd]
  df = df.set_index("time")
  
  f,ax = plt.subplots(figsize=(15,8))
  for c in ["OBS","MIX"]:
    ax.plot(np.arange(0,len(df)), df[c],marker="o", label=c)
  ax.set_ylim(0,1200)
  ax.set_xticks(np.arange(0,len(df)))
  ax.set_xlim(0,len(df))
  #--------
  bx = ax.twinx()
  bx.plot(np.arange(0,len(df)), df["DIFF"],marker="o",color="r", label="DIFF", lw="6")
  bx.axhline(y=0, color="k", alpha=0.7)
  bx.set_ylim(-600,600)
  bx.set_ylabel(r"DIFF Radiation [W/$m^2$]")
  #-----------
  ax.set_xticklabels(df.index, fontsize=12, rotation=45)
  ax.set_title(f"Solar Radiation [{dd}]")
  ax.set_ylabel(r"Solar Radiation [W/$m^2$]")
  # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
  ax.legend(loc = "upper left")
  bx.legend(loc = "upper right")
  f.savefig(f"{TMP}/tmp.png", bbox_inches="tight")

if __name__ == "__main__":
  for n_cate in [1,2]:
    main(n_cate)
  # resid_rad()
  