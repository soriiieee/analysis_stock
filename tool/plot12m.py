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

def plot12m(_df,x,y_list,_title, param=None):
    #init instance
    f, ax = plt.subplots(3,4,figsize=(16,12))
    ax = ax.flatten()

    for i,(df, title) in enumerate(zip(_df,_title)):
        # print(title, df.shape)
        # continue
        #datatime
        if df.empty:
            ax[i].set_visible(False)
        else:
            
            if df["time"].dtypes == object: df["time"] = pd.to_datetime(df["time"])
            #plot
            for j,y in enumerate(y_list):
                ax[i].scatter(df[x].values,df[y].values, s=2)
            
            #ax title
            ax[i].set_title(title, loc="left")
            if param["xlim"]:
                # print(param["xlim"])
                ax[i].set_xlim(param["xlim"])
            if param["ylim"]: 
                ax[i].set_ylim(param["ylim"])
            if param["xlabel"]: 
                ax[i].set_xlabel(param["xlabel"])
            if param["ylabel"]: 
                ax[i].set_ylabel(param["ylabel"])
            if param["xy_line"]: 
                ax[i].plot(np.arange(param["xlim"][1]),np.arange(param["xlim"][1]),color="k",lw=1)
                
            
    #axの間隔調整-> http://ailaby.com/subplots_adjust/
    plt.subplots_adjust(wspace=0.4, hspace=0.5)
    return f

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

