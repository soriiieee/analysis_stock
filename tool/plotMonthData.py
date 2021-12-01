# -*- coding: utf-8 -*-
#--------------------------------------------------------
#   DETAILES
#   program     :   getPlot
#   edit        :   yuichi sorimachi(20/01/06)
#   action      :   plot

#--------------------------------------------------------
#   module

import numpy as np
import pandas as pd
from matplotlib import rc
import matplotlib.pyplot as plt
cm = plt.cm.get_cmap("tab20")
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

def plotDayfor1Month(df,_col,vmin=0,vmax=1000,month=None,step=None,figtype="plot",title=False):
    #init instance
    f, ax = plt.subplots(5,7,figsize=(28,20))
    ax = ax.flatten()
    if df["time"].dtypes == object:
        df["time"] = pd.to_datetime(df["time"])
    df["day"] = df["time"].apply(lambda x: x.day)
    
    last_day = np.max(df["day"])

    for i ,day in enumerate(df["day"].unique()):
        df1 = df[df["day"]==day].reset_index(drop=True)
        # df1.index = [ idx % int(60 / timestep) for idx in df1.index]
        # stepsize=12*6
        df1["hhmm"] = df1["time"].apply(lambda x: x.strftime("%H%M"))
        list_dd = df1["hhmm"].values
        
        if figtype=="plot":
            for kk,col in enumerate(_col):
                ax[i].plot(np.arange(len(df1)),df1[col].values,label=col)
                # ax[i].plot(df1[col],label=col, color= cm.colors[kk], alpha=.7)
        elif figtype=="bar":
            col=_col[0]
            ax[i].bar(np.arange(len(df1)), df1[col],label=col)
        
        ax[i].set_xlim(0,len(df1))
        if month:
            ax[i].set_title(month+str(day).zfill(2), loc="left")
        if step:
            ax[i].set_title(month+str(day).zfill(2), loc="left")
            ax[i].set_xticks(np.arange(len(df1)))
            # ax[i].set_xticklabels(df1["hhmm"].values.tolist())
            st, ed = ax[i].get_xlim()
            # ax[i].xaxis.set_ticks(list_dd[0:len(df1):step])
            ax[i].xaxis.set_ticks(np.arange(st,ed,step))
        ax[i].set_ylim(vmin,vmax)
        ax[i].set_xlabel("time[0-48]")
        # ax[i].set_ylabel(r"$\displaystyle Solar-Rad[W/m^2] $")
        ax[i].set_ylabel(r"Solar-Rad[W/$m^2$]")
        # sys.exit()
    if title:
        f.suptitle(title,fontsize=20)
    
    #remoce cleaning
    # print(ax)
    # sys.exit()
    for i in range(last_day,5*7):
        # ax.remove(ax[i])
        # ax[i] = np.nan
        # ax[i].clear()
        ax[i].set_visible(False)
    
    #axの間隔調整-> http://ailaby.com/subplots_adjust/
    plt.subplots_adjust(wspace=0.4, hspace=0.5)
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

