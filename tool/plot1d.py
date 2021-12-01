# -*- coding: utf-8 -*-
#--------------------------------------------------------
#   DETAILES
#   program     :   plot1d
#   edit        :   yuichi sorimachi(21/07/14)
#   action      :   plot

#--------------------------------------------------------
#   module

import numpy as np
import pandas as pd
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

cm = plt.cm.get_cmap("tab20")
import matplotlib.colors as mcolors

import os
import sys

from datetime import datetime
now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")

#--------------------------------------------------------
# plt.rcParams['figure.figsize'] = [6.4,4.0]  # 図の縦横のサイズ([横(inch),縦(inch)])
# lauout -> https://qiita.com/aurueps/items/d04a3bb127e2d6e6c21b
# ref -> https://matplotlib.org/stable/api/matplotlib_configuration_api.html?highlight=rcparams#matplotlib.RcParams
fontsize=14
plt.rcParams['xtick.labelsize'] = fontsize        # 目盛りのフォントサイズ
plt.rcParams['ytick.labelsize'] = fontsize        # 目盛りのフォントサイズ
plt.rcParams['figure.subplot.wspace'] = 0.20 # 図が複数枚ある時の左右との余白
plt.rcParams['figure.subplot.hspace'] = 0.20 # 図が複数枚ある時の上下との余白
plt.rcParams['font.size'] = fontsize
plt.rcParams['lines.linewidth'] = 5
# plt.rcParams['text.usetex'] = True
# sys.exit()




# return me,rmse,nn
def plot1d_ec(df,_col,_mem_col=False,dd=False,vmin=0,vmax=1000,title=False,step=False,rotation=90):
    """
    2021.07.14
    df: "time"はdatetime
    _col: 表示するカラム
    _mem_col: mem_col
    dd = "20210714"のような8桁
    
    2021.08.30
    _colが無い場合の処理
    """
    # check !!! --------------
    if not dd:
        sys.exit("[ERROR!] please input dd !")
    
    if _col:
        if len(_col)<=10:
            _color = [ mcolors.rgb2hex(plt.cm.get_cmap('tab10')(i)) for i in range(10)]
        else:
            _color = [ mcolors.rgb2hex(plt.cm.get_cmap('tab20')(i)) for i in range(20)]
    else:
        pass
    # counting !!! --------------
    #init instance
    f, ax = plt.subplots(figsize=(12,8))
    # ax = ax.flatten()
    
    # _t = pd.date_range(start = f"{month}010000", periods = 31, freq="1D")
    # _dd = [ t.strftime("%Y%m%d") for t in _t]
    # _dd = [ t.day for t in _t]
    
    # if df["time"].dtypes == object:
    #     df["time"] = pd.to_datetime(df["time"])
    df["dd"] = df["time"].apply(lambda x: x.strftime("%Y%m%d"))
    
    # print(dd, type(dd))
    df = df[df["dd"]==dd]
    # print(df.head())
    # sys.exit()
    
    df = df[df["dd"]==dd]
    if df.shape[0] ==0:
        sys.exit(f"Not Founded this DD {dd} in DataFrame !")

    df["hhmm"] = df["time"].apply(lambda x: x.strftime("%H:%M"))
    list_dd = df["hhmm"].values
        
        # if figtype=="plot":
    if _col: #2021.08.30
        for kk,col in enumerate(_col):
            color = _color[kk]
            ax.plot(np.arange(len(df)),df[col].values,label=col,color=color,alpha=0.9,lw=5)
                # ax[i].plot(df1[col],label=col, color= cm.colors[kk], alpha=.7)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    # ax.legend(loc='upper right')
    if _mem_col:
        for k,col in enumerate(_mem_col):
            color = "gray"
            ax.plot(np.arange(len(df)),df[col].values,label=col,color=color,alpha=0.4,lw=1)
        
        
    ax.set_xlim(0,len(df))
    # ax[i].legend(ncol=3)で3行設定
    if title:
        ax.set_title(title, loc="left")
        # if step:
    
    if vmax:
        ax.set_ylim(vmin,vmax)
    ax.set_ylabel("日射量[W/m2]")
    ax.set_xlabel("時刻")
    if step:
        
        ax.set_xticks(np.arange(len(df)))
        # ax.set_xticklabels(df["hhmm"].values.tolist(),rotation=70)
        ax.set_xticklabels(df["hhmm"].values.tolist(), rotation=rotation)
        st, ed = ax.get_xlim()
        # ax[i].xaxis.set_ticks(list_dd[0:len(df1):step])
        ax.xaxis.set_ticks(np.arange(st,ed,step))
    
    #     #-dd loop ----------------
        
    # if title:
    #     f.suptitle(title,fontsize=20)
    # #axの間隔調整-> http://ailaby.com/subplots_adjust/
    # plt.subplots_adjust(wspace=0.4, hspace=0.5)
    return plt

if __name__ == "__main__":
    DIR="/home/griduser/work/sola8now_200507/tmp/0703_tmp/out"
    # 201912_unyo018.csv
    col="unyo018"
    input_path=f"{DIR}/201912_{col}.csv"
    df = pd.read_csv(input_path)
    df = df.replace(9999, np.nan)
    df["time"] = pd.to_datetime(df["time"].astype(str))
    # df[] = pd.to_datetime(df["time"].astype(str))

    print("start plotting")
    plotfigure = plotDayfor1Month(df,[col],title=os.path.basename(input_path),timestep=5)
    outdir = "/home/griduser/work/sola8now_200507/tmp/0703_tmp/png_month/"
    plotfigure.savefig(f"{outdir}/201912_{col}.png", bbox_inches ="tight")
    print("end plotting")

