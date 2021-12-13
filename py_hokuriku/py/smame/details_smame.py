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
#---
import matplotlib.colors as mcolors
_color = [ mcolors.rgb2hex(plt.cm.get_cmap('tab10')(i)) for i in range(10)]
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
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

sys.path.append('/home/ysorimachi/work/hokuriku/py')
from utils import *

SMAME="/work/ysorimachi/hokuriku/dat2/smame/dataset2"
SMAME_TBL="/home/ysorimachi/work/hokuriku/tbl/smame"

def dupli_code(cate="all"):
  _path = sorted(glob.glob(f"{SMAME}/{cate}_*.csv"))
  tbl = list_smame()
  num={}
  #----------------------------
  def check_duplicate(_code,tbl_code):
    """
    smame のうち記録はあるけど、tblに記載のない地点がいくつかある
    """
    s_code = set(_code)
    s_tbl_code = set(tbl_code)
    s_diff = s_tbl_code - s_code
    if len(s_diff) == 0:
      s_diff = s_code - s_tbl_code
      if len(s_diff) == 0:
        return None,"NaN"
      else:
        return list(s_diff),"smame"
    else:
      return list(s_diff),"tbl"
  #----------------------------
  
  for path in tqdm(_path):
    dd = os.path.basename(path).split("_")[1][:8]
    df = pd.read_csv(path)
    
    # df["duplicated"] = df.duplicated(subset=["code"])
    # print(df["duplicated"].sum())
    # sys.exit()
    _code = sorted(df["code"].values.tolist())
    n_data = len(_code)
    # print(dd,f"is {cate} num=",len(_code))
    tbl_code = sorted(tbl.loc[tbl["code"].isin(_code),"code"].values.tolist())
    tbl2 = tbl.loc[tbl["code"].isin(_code),:]
    mean,std,min0,max0 = tbl2.describe()["max[kW]"][["mean","std","min","max"]].values
    
    # print(len(_code),len(tbl_code))
    
    dupli_list,many = check_duplicate(_code,tbl_code)
    try:
      n_dupli = len(dupli_list)
      dupli_str = "/".join(dupli_list)
    except:
      n_dupli = 0
      dupli_str=""
      
    num[dd] = [n_data,many,n_dupli,dupli_str,mean,std,min0,max0]
    
  df = pd.DataFrame(num).T
  df.columns = ["n_data","many","n_dupli","dupli_str","mean","std","min","max"]
  df.index.name = "dd"
  df.to_csv(f"{SMAME_TBL}/{cate}_detail.csv")
  return

def check_dupli(cate="all"):
  path = f"{SMAME_TBL}/{cate}_detail.csv"
  df = pd.read_csv(path)
  df = df[df["dupli_str"] != ""]
  df = df.dropna(subset=["dupli_str"])
  # print(df.isnull().sum())
  # sys.exit()
  list_dupli = df["dupli_str"].values.tolist()
  list_dupli2 = [ set(c.split("/")) for c in list_dupli]
  # sys.exit()
  # 和集合の作り方 ->  https://note.nkmk.me/python-set/
  s1 = list_dupli2[0]
  for s in tqdm(list_dupli2[1:]):
    s1.union(s)
  
  s1 = sorted(list(s1))
  
  tbl = list_smame()
  
  for code in s1:
    tmp = tbl[tbl["code"]==code]
    print(cate,code ,"->",tmp.shape)
  # sys.exit()
  if 0: #csv
    df = pd.DataFrame()
    df["code"] = s1
    df.to_csv(f"{SMAME_TBL}/tmp/{cate}_not_in_tbl.csv")
  if 0:
    with open(f"{SMAME_TBL}/tmp/{cate}_not_in_tbl.txt", "w") as f:
      for i ,code in enumerate(s1):
        i1 = str(i+1).zfill(4)
        text = f"{i1} {code}\n"
        f.write(text)
  return

def count_smame():
  OUTD="/home/ysorimachi/work/hokuriku/out/smame/detail/count"
  _mm = loop_month()
  _dd = [ f"{mm}25" for mm in _mm]
  
  num = {}
  for dd in _dd:
    path = f"{SMAME}/all_{dd}.csv"
    df = pd.read_csv(path)
    n_all = df.shape[0]
    path = f"{SMAME}/surplus_{dd}.csv"
    df = pd.read_csv(path)
    n_surplus = df.shape[0]
    
    num[dd] = [n_all,n_surplus]
  
  df = pd.DataFrame(num).T
  df.index = _mm
  df.columns = ["all","surplus"]
  df.index.name = "month"
  df.to_csv(f"{OUTD}/count.csv")
  return

def dist_smame():
  """
  date: 2021.07.26
  date: 2021.09.01 : update list
  """
  OUTD="/home/ysorimachi/work/hokuriku/out/smame/detail/count"
  _mm = loop_month()
  # _dd = [ f"{mm}25" for mm in _mm] #2021.07.26
  _dd = [ '20190425' ,"20200425"] #2021.09.01
  
  tbl = list_smame()
  for i,cate in enumerate(["all","surplus"]):
    #-loop---
    
    _bins=[]
    if cate == "all":
      bins_range= [0,5,10,15,20,25,30,35,40,45,50]
    if cate == "surplus":
      bins_range= [0,1,3,5,7,9,11,13,15,17,50]
      
    for dd in _dd:
      path = f"{SMAME}/{cate}_{dd}.csv"
      df = pd.read_csv(path)
      _code = df["code"].values.tolist()
      tb = tbl.loc[tbl["code"].isin(_code),:]
      
      max_list = tb["max[kW]"].values.tolist()
      df = pd.cut(max_list, bins=bins_range).value_counts() / len(max_list)
      _bins.append(df)
    
    df = pd.concat(_bins,axis=1)
    df = df.reset_index()
    name = "カテゴリ[kW]"
    df.columns = [name] + [ dd[:4]+ "年"+ dd[4:6] +"月" for dd in _dd]
    df= df.set_index(name)
    bins_range.remove(0)
    df.index = [ f"~{v}" for v in bins_range]
    # print(df.head())
    # sys.exit()
    
    df.to_csv(f"{OUTD}/hist_{cate}.csv",encoding="shift-jis")
    
    if 1:
      rcParams["font.size"] = 18
      w=0.4
      f,ax = plt.subplots(figsize=(12,7))
      for i,c in enumerate(df.columns):
        ax.bar(np.arange(len(df))+ w*i,df[c],width=w, color=_color[i], label=c)
      
      ax.legend(loc="upper left")
      # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
      ax.set_xlabel("受給最大電力[kW]")
      ax.set_xticks(np.arange(len(df)))
      ax.set_xticklabels(df.index, rotation=0)
      ax.set_ylabel("設備数の相対度数(割合)")
      ax.set_title("受給最大電力階級別設備数")
      
      f.savefig(f"{OUTD}/hist_{cate}.png",bbox_icnhes="tight")
      # f.savefig(f"{OUTD}/hist_{cate}.png")
    print(datetime.now(), "[END]", cate)
    
      # print(np.max(max_list), np.min(max_list), np.mean(max_list), np.std(max_list))
      # print(bins)
      # sys.exit()
      # sns.distplot(tb["max[kW]"],ax=ax[i], label=dd[:6],bins=30)
    #-setting---
  #   ax[i].set_title(cate)
  #   ax[i].set_xlim(0,50)
  #   ax[i].legend()
    
  #   # min0= np.round(np.min(tb["max[kW]"]),1)
  #   # ave0= np.round(np.mean(tb["max[kW]"]),1)
  #   # max0= np.round(np.max(tb["max[kW]"]),1)
  #   # ymin,ymax = ax[i].get_ylim()
  #   ax[i].text(10,0.6*ymax,f"min={min0}[kW]\nave={ave0}[kW]\nmax={max0}[kW]", fontsize=16)
  #   # sys.exit()
  # plt.tight_layout()
  # f.savefig(f"{OUTD}/hist.png",bbox_icnhes="tight")
  return
      
def mean_smame():
  OUTD="/home/ysorimachi/work/hokuriku/out/smame/detail/count"
  _mm = loop_month()
  _dd = [ f"{mm}25" for mm in _mm]
  tbl = list_smame()
  f,ax = plt.subplots(2,1,figsize=(10,10))
  for i,cate in enumerate(["all","surplus"]):
    #-loop---
    # for dd in _dd[::3]:
    v = {}
    for dd in _dd:
      path = f"{SMAME}/{cate}_{dd}.csv"
      df = pd.read_csv(path)
      _code = df["code"].values.tolist()
      tb = tbl.loc[tbl["code"].isin(_code),:]
      
      min0= np.round(np.min(tb["max[kW]"]),2)
      max0= np.round(np.max(tb["max[kW]"]),2)
      ave0= np.round(np.mean(tb["max[kW]"]),2)
      std0= np.round(np.std(tb["max[kW]"]),2)
      
      v[dd[:6]] = [min0,max0,ave0,std0]
    
    df = pd.DataFrame(v).T
    df.columns = ["min","max","mean","std"]
    df.index.name = "month"
    df.to_csv(f"{OUTD}/details_{cate}.csv")
    # sys.exit()
  return



if __name__ == "__main__":
  
  if 0:
    # check dupli --------------
    for cate in ["surplus","all"]:
      # dupli_code(cate=cate) #"重複確認用"
      check_dupli(cate=cate) #重複リスト作成
  
  if 1:
    # count_smame() #個数の把握 -> 21.07.22
    # mean_smame() #平均の把握 -> 21.07.26
    dist_smame()
    #--------------