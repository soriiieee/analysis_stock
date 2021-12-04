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
import matplotlib.colors as mcolors
_color = [ mcolors.rgb2hex(plt.cm.get_cmap('tab10')(i)) for i in range(10)]

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

def plot_multi_rad(df,_dd,_col,_sub_col=None,vmin=0,vmax=1000, return_ax=False):
    #init instance ---------
    if len(_dd) == 12:
        f, ax = plt.subplots(3,4,figsize=(28,20))
    elif len(_dd) == 20:
        f, ax = plt.subplots(4,5,figsize=(28,20))
    elif len(_dd) == 30:
        f, ax = plt.subplots(5,6,figsize=(28,20))
    elif len(_dd) == 35:
        f, ax = plt.subplots(5,7,figsize=(28,20))
    elif len(_dd) == 56:
        f, ax = plt.subplots(7,8,figsize=(28,20))
    else:
        sys.exit("please input 12,20,30,35,56 !")
    ax = ax.flatten()

    #time to datetime
    if df["time"].dtypes == object:
        df["time"] = pd.to_datetime(df["time"])
    
    df["day"] = df["time"].apply(lambda x: x.strftime("%Y%m%d"))
    # n_last = len(df["day"].unique())-1

    for i ,day in enumerate(_dd):
        df1 = df[df["day"]==day].reset_index(drop=True)
        # df1.index = [ idx % int(60 / timestep) for idx in df1.index]
        # stepsize=12*6
        df1["hhmm"] = df1["time"].apply(lambda x: x.strftime("%H"))
        list_dd = df1["hhmm"].values
        
        for kk,col in enumerate(_col):
            color = _color[kk]
            ax[i].plot(np.arange(len(df1)),df1[col].values,label=col,color=color,alpha=0.8,lw=5)
                # ax[i].plot(df1[col],label=col, color= cm.colors[kk], alpha=.7)
        
        """ 2021.07.18 同じax[i]に乗せる場合"""
        # if _sub_col:
        #     for k,col in enumerate(_sub_col):
        #         color = "gray"
        #         ax[i].plot(np.arange(len(df1)),df1[col].values,label=col,color=color,alpha=0.3,lw=1)
                
        """ 2021.07.26 右bxに乗せる場合"""
        if _sub_col:
            bx= ax[i].twinx()
            for k,col in enumerate(_sub_col):
                color = "gray"
                bx.plot(np.arange(len(df1)),df1[col].values,color="k",alpha=0.3,lw=1)
                # bx.fill_between(x = np.arange(len(df1)),
                #         y1 = df1[col].values,
                #         y2 = np.zeros(len(df1)),color="green",alpha=0.3)
            
            snow_max = np.nanmax(df1[col]) 
            bx.set_ylim(0,snow_max)
        """------------------------------- """ 
        
        ax[i].set_xlim(0,len(df1))
        if 1:
            ax[i].set_xticks(np.arange(len(df1)))
            ax[i].set_xticklabels(df1["hhmm"].values.tolist(), rotation=0)
            st, ed = ax[i].get_xlim()
            # ax[i].xaxis.set_ticks(list_dd[0:len(df1):step])
            ax[i].xaxis.set_ticks(np.arange(st,ed,6))
        ax[i].set_title(day, loc="left")
        ax[i].tick_params(labelbottom=1,labelleft=1,labelright=False,labeltop=False)
        
        if i%5==4:
            ax[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
            # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18)
        # if i == n_last:
        #     """2021.06.30 sorimachi add"""
        #     ax[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18)
        if vmax:
            ax[i].set_ylim(vmin,vmax)
        # ax[i].set_xlabel("time[0-48]")
        # # ax[i].set_ylabel(r"$\displaystyle Solar-Rad[W/m^2] $")
        # ax[i].set_ylabel(r"Solar-Rad[W/$m^2$]")
        # sys.exit()
    # if title:
    #     f.suptitle(title,fontsize=20)
    
    #axの間隔調整-> http://ailaby.com/subplots_adjust/
    plt.subplots_adjust(wspace=0.4, hspace=0.5)
    if return_ax:
        return f,ax
    else:
        return f



def plot1m_ec(df,_col,_mem_col=False,month=False,vmin=0,vmax=1000,title=False):
    """
    2021.07.14
    df: "time"はdatetime
    _col: 表示するカラム
    _mem_col: mem_col
    
    """
    # check !!! --------------
    if not month:
        sys.exit("[ERROR!] please input month !")
    
    # counting !!! --------------
    #init instance
    f, ax = plt.subplots(5,7,figsize=(28,20))
    ax = ax.flatten()
    
    _t = pd.date_range(start = f"{month}010000", periods = 31, freq="1D")
    # _dd = [ t.strftime("%Y%m%d") for t in _t]
    _dd = [ t.day for t in _t]
    
    # if df["time"].dtypes == object:
    #     df["time"] = pd.to_datetime(df["time"])
    df["day"] = df["time"].apply(lambda x: x.day)
    
    last_day = np.max(df["day"])
    n_last = len(df["day"].unique())-1

    for i ,day in enumerate(_dd):
        df1 = df[df["day"]==day].reset_index(drop=True)
        # print(df1.shape)
        # continue
        # sys.exit()
        #DataFrameが0じゃなければ、描画の開始を行う予定
        if df1.shape[0] !=0:
                
            # df1.index = [ idx % int(60 / timestep) for idx in df1.index]
            # stepsize=12*6
            df1["hhmm"] = df1["time"].apply(lambda x: x.strftime("%H%M"))
            list_dd = df1["hhmm"].values
            
            # if figtype=="plot":
            for kk,col in enumerate(_col):
                color = _color[kk]
                ax[i].plot(np.arange(len(df1)),df1[col].values,label=col,color=color,alpha=0.8,lw=5)
                    # ax[i].plot(df1[col],label=col, color= cm.colors[kk], alpha=.7)
            
            if _mem_col:
                for k,col in enumerate(_mem_col):
                    color = "gray"
                    ax[i].plot(np.arange(len(df1)),df1[col].values,label=col,color=color,alpha=0.3,lw=1)
            
            
            ax[i].set_xlim(0,len(df1))
            ax[i].set_ylim(vmin,vmax)
            if month:
                ax[i].set_title(month+str(day).zfill(2), loc="left")
            # if step:
            #     ax[i].set_title(month+str(day).zfill(2), loc="left")
            #     ax[i].set_xticks(np.arange(len(df1)))
            #     # ax[i].set_xticklabels(df1["hhmm"].values.tolist())
            #     st, ed = ax[i].get_xlim()
            #     # ax[i].xaxis.set_ticks(list_dd[0:len(df1):step])
            #     ax[i].xaxis.set_ticks(np.arange(st,ed,step))
            
            # if i == n_last:
            #     """2021.06.30 sorimachi add"""
            #     ax[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18)
            # ax[i].set_ylim(vmin,vmax)
            # ax[i].set_xlabel("time[0-48]")
            # # ax[i].set_ylabel(r"$\displaystyle Solar-Rad[W/m^2] $")
            # ax[i].set_ylabel(r"Solar-Rad[W/$m^2$]")
        # sys.exit()
        else:
            # ax[i].set_visible(False)
            pass
        
        #-dd loop ----------------
        
    if title:
        f.suptitle(title,fontsize=20)
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

