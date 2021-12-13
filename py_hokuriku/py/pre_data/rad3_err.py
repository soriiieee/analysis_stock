# -*- coding: utf-8 -*-
# when   : 2021.07.15
# who : [sori-machi]
# what : 北陸技研の記録についてセットアップする(異常値等の確認)
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
from getErrorValues import me,rmse,mape,r2 #(x,y)
from convSokuhouData import conv_sfc #(df, ave=minutes,hour)
#amedas relate 2020.02,04 making...
from tool_AMeDaS import code2name, name2code
from tool_110570 import get_110570,open_110570
from tool_100571 import get_100571,open_100571
#(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
import subprocess

from plot1m import plot1m, plot1m_ec, plot1m_2axis #(df,_col,_mem_col=False,month=False,vmin=0,vmax=1000,title=False)
from plot1d import plot1d_ec #title=False)
from utils import *
from sklearn.linear_model import LinearRegression

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

# outd ='/home/griduser/work/sori-py2/deep/out/0924_01'
# os.makedirs(outd, exist_ok=True)
# subprocess.run('rm -f *.png', cwd=outd, shell=True)

RAD_DAT="/home/ysorimachi/work/hokuriku/dat/rad/re_get"
OUT_1MIN = "/work/ysorimachi/hokuriku/dat2/rad/obs/1min"


#--------------------
def r2_base():
  """
  2021.07.10 : init
  2021.08.10 : re-write
  """
  OUTD="/work/ysorimachi/hokuriku/dat2/rad/per_code"
  ERR_D="/home/ysorimachi/work/hokuriku/dat/rad/r2"
  mem_col = ["kans001","kans002",'unyo001', 'unyo002', 'unyo003', 'unyo004', 'unyo005',
       'unyo006', 'unyo007', 'unyo008', 'unyo009', 'unyo010', 'unyo011',
       'unyo012', 'unyo013', 'unyo014', 'unyo015','unyo017',
       'unyo018']
  # mem_col = ["mean"]
  def ave_rad5to30(df):
    for c in ["obs","8Now0"]:
      df[c] = df[c].rolling(6).mean()
    df = df.iloc[::6,:]
    return df
  #-------------------------------------
  if 0:
    """ pre data2(obs,8Now0) """
    for code in mem_col:
      _rad = []
      for month in loop_month()[:12]:
        rad = concat_2rad(code,month)
        _rad.append(rad)
      
      df = pd.concat(_rad,axis=0)
      df.to_csv(f"{OUTD}/rad_{code}.csv", index=False)
      print(datetime.now(), "[END]", code)
  #-------------------------------------
  
  isPlot=True
  f,ax = plt.subplots(4,5,figsize=(25,20))
  ax = ax.flatten()
  
  err_hash = {}
  for i,code in enumerate(mem_col + ["mean"]):
  # for i,code in enumerate(["mean"]):
    
    path = f"{OUTD}/rad_{code}.csv"
    df = pd.read_csv(path)
    # # print(path)
    # print(df.head())
    # print(df.describe())
    # sys.exit()
    df = df.replace(9999,np.nan).replace("NaN",np.nan)
    df["time"] = pd.to_datetime(df["time"])
    df["mm"] = df["time"].apply(lambda x: x.strftime("%Y%m"))
    # df = ave_rad5to30(df)
    df = df.dropna()
    df["yy"] = df["month"].apply(lambda x: "2020" if x>=202004 else "2019")
    
    def get_err(df):
      me_score = me(df["obs"],df["8Now0"])
      rmse_score = rmse(df["obs"],df["8Now0"])
      r2_score = np.round(r2(df["obs"],df["8Now0"]),3)
      return [me_score,rmse_score,r2_score]
    #plot ----------
    if isPlot:
      
      for yy in ["2019","2020"]:
        
        tmp = df[df["yy"]==yy]
        _err =  get_err(tmp)
        err_hash[f"{code}_{yy}"] = _err
        r2_score = _err[2]
        ax[i].scatter(tmp["obs"],tmp["8Now0"],s=1, label = f"R2({yy}):{r2_score}")
      ax[i].plot(np.arange(1200),np.arange(1200), lw=1,color="k")
      ax[i].set_xlim(0,1200)
      ax[i].set_ylim(0,1200)
      ax[i].legend()
      name = rad_code2name(code)
      N = df.shape[0]
      ax[i].set_title(f"{code}({name[:4]})-N={N}")
    #---------------
    print("end",code)
  
  if isPlot:
    plt.subplots_adjust(wspace=0.4, hspace=0.5)
    f.savefig(f"{ERR_D}/r2_score.png", bbox_inches="tight")
    print(f"{ERR_D}/r2_score.png")
  
  df = pd.DataFrame(err_hash).T
  df.columns = ["me","rmse","r2"]
  df.index.name = "code"
  df.to_csv(f"{ERR_D}/err_all_term.csv")
  print(df.head())
  sys.exit()


def r2_mm():
  """
  2021.07.15 最初に作成した状況
  2021.08.10 荒井さんへ報告用に修正
  2021.09.08 荒井さんへ報告用に修正/ 
  2021.09.11 荒井さんへ報告用に修正
  """
  OUTD="/work/ysorimachi/hokuriku/dat2/rad/per_code"
  ERR_D="/home/ysorimachi/work/hokuriku/dat/rad/r2"
  mem_col = ["kans001","kans002",'unyo001', 'unyo002', 'unyo003', 'unyo004', 'unyo005',
       'unyo006', 'unyo007', 'unyo008', 'unyo009', 'unyo010', 'unyo011',
       'unyo012', 'unyo013', 'unyo014', 'unyo015','unyo017',
       'unyo018']
  
  
  # mem_col = ["mean"]
  def ave_rad5to30(df):
    for c in ["obs","8Now0"]:
      df[c] = df[c].rolling(6).mean()
    df = df.iloc[::6,:]
    return df
  
  isPlot=True
  _code =["unyo006"]
  err_hash = {}
  # _mm = ['201904','201905','201906','201907','201908','201909','201910','201911','201912','202001','202002','202003']
  _mm = loop_month()[12:]
  _mm19 = loop_month()[:12]
  # sys.exit()
  # for i,code in enumerate(mem_col + ["mean"]):
  for i,code in enumerate(_code):
    f,ax = plt.subplots(3,4,figsize=(20,15))
    ax = ax.flatten()
    
    df = pd.read_csv(f"{OUTD}/rad_{code}.csv")
    df = df.replace(9999,np.nan).replace("NaN",np.nan)
    df["time"] = pd.to_datetime(df["time"])
    df["mm"] = df["time"].apply(lambda x: x.strftime("%Y%m"))
    df["dd"] = df["time"].apply(lambda x: x.strftime("%Y%m%d"))
    df.loc[df["dd"]=="20200324","8Now0"] = np.nan
    df.loc[df["dd"]=="20210324","8Now0"] = np.nan
    # df = ave_rad5to30(df)
    # print(df.head())
    # sys.exit()
    
    
    err_hash={}
    for j,(mm,mm19) in enumerate(zip(_mm,_mm19)):
      tmp = df[df["mm"]==mm]
      tmp19 = df[df["mm"]==mm19]
      
      # clensing 2021.09.08
      tmp = tmp[(tmp["obs"]>10)&(tmp["8Now0"]>10)]
      tmp19 = tmp19[(tmp19["obs"]>10)&(tmp19["8Now0"]>10)]
      
      lr = LinearRegression(fit_intercept=False).fit(tmp["obs"].values.reshape(-1,1),tmp["8Now0"])
      y_pred = lr.predict(np.arange(1200).reshape(-1,1))
      
      a = np.round(lr.coef_[0],3)
      me_score = me(tmp["obs"],tmp["8Now0"])
      rmse_score = rmse(tmp["obs"],tmp["8Now0"])
      
      me_score19 = me(tmp19["obs"],tmp19["8Now0"])
      rmse_score19 = rmse(tmp19["obs"],tmp19["8Now0"])
      # r2_score = np.round(r2(tmp["obs"],tmp["8Now0"]),3)
      
      # err_hash[mm] = [me_score,rmse_score,r2_score,a]
      # err_hash[mm] = [me_score,rmse_score,r2_score,a]
      err_hash[mm] = [me_score,rmse_score,9999,9999]
      err_hash[mm19] = [me_score19,rmse_score19,9999,9999]
      #plot ----------
      if 1:
        ax[j].scatter(tmp19["obs"],tmp19["8Now0"],s=7,color="green", label="2019年度", alpha=0.7)
        # ax[j].scatter(tmp["obs"],tmp["8Now0"],s=7, label = f"R2={r2_score}")
        ax[j].scatter(tmp["obs"],tmp["8Now0"],s=7, label = "2020年度")
        # ax[j].scatter(tmp19["obs"],tmp19["8Now0"],s=2,color="gray", label="2019")
        ax[j].plot(np.arange(1200),np.arange(1200), lw=1,color="k")
        ax[j].plot(np.arange(1200),y_pred, lw=1,color="r", label=f"y={a}x")
        ax[j].set_xlim(0,1200)
        ax[j].set_ylim(0,1200)
        
        ax[j].set_xlabel("地上運用[W/m2]")
        ax[j].set_ylabel("8Now0衛星[W/m2]")
        
        ax[j].legend()
        name = rad_code2name(code)
        N = tmp.shape[0]
        ax[j].set_title(f"{code}({name[:5]})-{mm}")
        
    plt.subplots_adjust(wspace=0.4, hspace=0.5)
    f.savefig(f"{ERR_D}/code/rad_mm_{code}.png", bbox_inches="tight")
    plt.close()
    print(f"{ERR_D}/code")
    
    df = pd.DataFrame(err_hash).T
    df.columns = ["me","rmse","r2","a"]
    df.index.name = "mm"
    # print(df.head())
    # sys.exit()
    df.to_csv(f"{ERR_D}/code/err_{code}.csv")
    print(datetime.now(), "[end]", code)
  return
    # sys.exit()

def load_2rad(code,month):
  OUTD="/work/ysorimachi/hokuriku/dat2/rad/per_code"
  df = pd.read_csv(f"{OUTD}/rad_{code}.csv")
  df = df.replace(9999,np.nan).replace("NaN",np.nan)
  df["time"] = pd.to_datetime(df["time"])
  df["mm"] = df["time"].apply(lambda x: x.strftime("%Y%m"))
  df = df[df["mm"]==month]
  return df

def get_snowdepth(month,scode):
  """
  2021.07.27 sorimachi making
  """
  local = "/work/ysorimachi/hokuriku/dat2/snow_sfc/origin"
  path = f"{local}/sfc_10minh_{month}_{scode}.csv"
  df = pd.read_csv(path)
  df = conv_sfc(df,ave=30)
  df = df.replace(9999,np.nan)
  df["snowDepth"] = df["snowDepth"].fillna(method="pad")
  return df[["time","snowDepth"]]

def check_rad_ts(code,month, is_csv = None):
  TS_OUT="/home/ysorimachi/work/hokuriku/dat/rad/r2/rad"
  df = load_2rad(code,month)
  df = df.drop(["mi","month","mm"], axis=1)
  s1 = get_snowdepth(month, "47607")
  s2 = get_snowdepth(month, "47616")
  df = df.merge(s1,on="time", how="inner")
  df = df.merge(s2,on="time", how="inner")
  df.columns = ["time","obs","8Now0","snw_toyama","snw_fukui"]
  
  if is_csv:
    return df
  
  # f = plot1m_2axis(df,_col=["obs","8Now0"],_sub_col=["snw_toyama"],month=month,_ylim=[0,1000,0,150],title=False,step=6)
  f = plot1m_2axis(df,_col=["obs","8Now0"],_sub_col=None,month=month,_ylim=[0,1000,0,150],title=False,step=6)
  plt.subplots_adjust(wspace=0.6, hspace=0.6)
  f.savefig(f"{TS_OUT}/{code}_{month}.png", bbox_inches="tight")
  print(TS_OUT, month)
  plt.close()
  return

def plot_1day(dd):
  TS_OUT="/home/ysorimachi/work/hokuriku/dat/rad/r2/rad"
  
  month = dd[:6]
  code = "mean"
  df = check_rad_ts(code,month, is_csv =True)
  df["dd"] = df["time"].apply(lambda x: x.strftime("%Y%m%d"))
  df["hh"] = df["time"].apply(lambda x: x.hour)
  df = df[df["dd"]==dd]
  df = df[(df["hh"]>=5)&(df["hh"]<=20)]
  # print(df.shape)
  # print(df.head())
  # sys.exit()
  
  df["diff"] = df["obs"] - df["8Now0"]
  max_diff= np.round(np.max(np.abs(df["diff"])),1)
  
  f,ax = plt.subplots(figsize=(12,6))
  rcParams["font.size"] = 18
  for c in [ "obs","8Now0"]:
    ax.plot(np.arange(len(df)),df[c], label=c)
  ax.set_ylim(0,1000)
  ax.legend(loc="upper left")
  # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
  ax.set_xlabel("時刻[-]")
  ax.set_ylabel("エリア平均日射量[W/m2]")
  ax.set_xticks(np.arange(len(df)))
  
  ax.text(20,900, f"最大誤差: {max_diff}[W/m2]")
  _t = pd.date_range(start=f"{dd}0500", periods=df.shape[0],freq="30T")
  _time = [ t.strftime("%H:%M") for t in _t]
  # _time = [ t.strftime("%H:%M") for t in df["time"].values]
  ax.set_xticklabels(_time, rotation=80)
  ax.set_title(f"日射量({dd}[JST])")
  
  f.savefig(f"{TS_OUT}/dd/rad_{dd}.png",bbox_icnhes="tight")
  print(f"{TS_OUT}/dd")
  return 
  
def r2_mm_v2():
  """
  2021.09.10 荒井さんへ報告用に修正
  """
  OUTD="/work/ysorimachi/hokuriku/dat2/rad/per_code"
  ERR_D="/home/ysorimachi/work/hokuriku/dat/rad/r2"
  mem_col = ["kans001","kans002",'unyo001', 'unyo002', 'unyo003', 'unyo004', 'unyo005',
       'unyo006', 'unyo007', 'unyo008', 'unyo009', 'unyo010', 'unyo011',
       'unyo012', 'unyo013', 'unyo014', 'unyo015','unyo017',
       'unyo018',"mean"]
  def cut_time(df,time_range = [6,18]):
    st,ed = time_range
    return df[(df["hh"]>=st)&(df["hh"]<=ed)]

  def clensing_210910(df,code,_col):
    df["dd"] = df["time"].apply(lambda x: x.strftime("%Y%m%d"))
    # remove 21.09.10
    if code=="unyo018":
      df.loc[df["dd"]=="20200518","obs"] = np.nan
    if code=="unyo015":
      df.loc[df["dd"]=="20201215","obs"] = np.nan
      df.loc[df["dd"]=="20201216","obs"] = np.nan
    if code=="unyo002":
      df.loc[df["dd"]=="20210224","obs"] = np.nan
      df.loc[df["dd"]=="20210224","obs"] = np.nan

    # 除外ルール　->　8Now0　メンテナンス？
    df.loc[df["dd"]=="20200324","8Now0"] = np.nan
    df.loc[df["dd"]=="20210324","8Now0"] = np.nan
    for c in _col:
      df[c] = df[c].apply(lambda x: isFloat(x))
    df = df.drop(["dd"],axis=1)
    return df

  

  # mem_col = ["mean"]
  def ave_rad5to30(df):
    for c in ["obs","8Now0"]:
      df[c] = df[c].rolling(6).mean()
    df = df.iloc[::6,:]
    return df
  
  isPlot=True
  # _code =["mean"]
  err_hash = {}
  # _mm = ['201904','201905','201906','201907','201908','201909','201910','201911','201912','202001','202002','202003']
  _mm = loop_month()
  _mm2 = [ m[4:6] for m in _mm]
  # sys.exit()
  # for i,code in enumerate(mem_col + ["mean"]):
  f,ax = plt.subplots(4,5,figsize=(20,15))
  ax = ax.flatten()
  for i,code in enumerate(mem_col):
    # f,ax = plt.subplots(3,4,figsize=(20,15))
    # ax = ax.flatten()
    # code="unyo002"
    if 1:
      #--------- ERR MAKING -----------------# 
      df = pd.read_csv(f"{OUTD}/rad_{code}.csv")
      df = df.replace(9999,np.nan).replace("NaN",np.nan)
      df["time"] = pd.to_datetime(df["time"])
      df["hh"] = df["time"].apply(lambda x: x.hour)
      
      # clensing ----------
      # df = cut_time(df,time_range = [6,18])
      df = clensing_210910(df,code,["obs","8Now0"]) # utils.py から
      df = df.dropna()
      df = df[df["obs"]>0]
      
      # from plot1m import plot1m
      # mm = "202103"
      # df =df[df["month"]==int(mm)]
      # f = plot1m(df,_col=["obs","8Now0"],vmin=0,vmax=1200,month=mm,step=None,figtype="plot",title=False)
      # f.savefig(f"/home/ysorimachi/work/hokuriku/dat/rad/r2/per_mm/debug/{code}_{mm}.png", bbox_icnhes="tight")
      # # df["diff"] = np.abs(df["obs"]-df["8Now0"])
      # # print(df.sort_values("diff",ascending=False).head(20))
      # # print(df.describe().T["max"])
      # sys.exit()

      err_hash={}
      for j,mm in enumerate(_mm):
        tmp = df[df["month"]==int(mm)]
        # clensing 2021.09.08
        # tmp = tmp[(tmp["obs"]>10)&(tmp["8Now0"]>10)]

        #err ----
        e1 = me(tmp["obs"],tmp["8Now0"])
        e2 = rmse(tmp["obs"],tmp["8Now0"])
        tmp = cut_time(tmp,time_range = [9,15])
        e3 = mape(tmp["obs"],tmp["8Now0"])
        err_hash[mm] = [e1,e2,e3]
        
      df = pd.DataFrame(err_hash).T
      df.index.name ="month"
      df.columns = [f"ME_{code}",f"RMSE_{code}",f"MAPE_{code}"]
      df.to_csv(f"{ERR_D}/per_mm/{code}.csv")
      #--------- ERR MAKING -----------------#
    
    if 0:
      #--------- PNG MAKING -----------------#
      err_name,vmin,vmax="ME",-100,100
      # err_name,vmin,vmax="RMSE",0,200
      # err_name,vmin,vmax="MAPE",-2,3
      
      col = f"{err_name}_{code}"
      path = f"{ERR_D}/per_mm/{code}.csv"
      df = pd.read_csv(path)
      
      if err_name == "MAPE":
        ax[i].bar(np.arange(len(df)),np.log10(df[col]))
      else:
        ax[i].bar(np.arange(len(df)),df[col].values)
        
      ax[i].set_xlabel("month")
      ax[i].set_xlabel("month")
      #----------
      step=3
      ax[i].set_xlim(0,len(df))
      ax[i].set_xticks(np.arange(len(df)))
      st, ed = ax[i].get_xlim()
      ax[i].xaxis.set_ticks(np.arange(int(st), int(ed),step))
      ax[i].set_xticklabels(_mm2[::step], fontsize=12)
      #----------
      ax[i].set_xlabel("month")
      ax[i].set_ylabel(f"{err_name}[W/m2]")
      ax[i].set_ylim(vmin,vmax)
      ax[i].set_title(f"{code}({rad_code2name(code)})")
      ax[i].axvline(x=11.5, color="k", lw=1)
      ax[i].text(1,0.5*vmax, "2019")
      ax[i].text(1+11,0.5*vmax, "2020")
      #--------- PNG MAKING -----------------#
    print(datetime.now(), "[END]", code)
  
  try:
    plt.subplots_adjust(wspace=0.4, hspace=0.5)
    f.savefig(f"{ERR_D}/per_mm/PNG_{err_name}.png",bbox_inches="tight")
  except:
    print("Not Making Plot ...")


if __name__ =="__main__":
  if 0:
    # 官署結合の為仕切り直し/地上観測点も同時取り込み
    pre_convert_rad()

  if 1:
    for month in ["202005","202006","202009"]:
      check_rad_ts(code="unyo006",month=month) #month
    sys.exit()
    
    _ini_j = [ "20210111","20210120","20210125"]
    for dd in _ini_j:
      plot_1day(dd=dd)
  
  if 0:
    # r2_base() #2021.08.10
    r2_mm() #2021.08.10/09.08(update) -> 2019/2020 code別のscatter
    # r2_mm_v2() #09.10(update)
    