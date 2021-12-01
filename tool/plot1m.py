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
def plot1m(df,_col,vmin=0,vmax=1000,month=None,step=None,figtype="plot",title=False):
    #init instance
    f, ax = plt.subplots(5,7,figsize=(28,20))
    ax = ax.flatten()
    if df["time"].dtypes == object:
        df["time"] = pd.to_datetime(df["time"])
    df["day"] = df["time"].apply(lambda x: x.day)
    
    last_day = np.max(df["day"])
    n_last = len(df["day"].unique())-1

    for i ,day in enumerate(df["day"].unique()):
        df1 = df[df["day"]==day].reset_index(drop=True)
        # df1.index = [ idx % int(60 / timestep) for idx in df1.index]
        # stepsize=12*6
        # df1["hhmm"] = df1["time"].apply(lambda x: x.strftime("%H:%M"))
        df1["hhmm"] = df1["time"].apply(lambda x: int(x.strftime("%H")))
        list_dd = df1["hhmm"].values
        
        if figtype=="plot":
            for kk,col in enumerate(_col):
                # ax[i].plot(np.arange(len(df1)),df1[col].values,label=col)
                # ax[i].plot(np.arange(len(df1)),df1[col].values,label=col,lw=1,alpha=0.8)
                ax[i].plot(np.arange(len(df1)),df1[col].values,label=col,lw=4,alpha=1)
                # ax[i].plot(df1[col],label=col, color= cm.colors[kk], alpha=.7)
        elif figtype=="bar":
            col=_col[0]
            ax[i].bar(np.arange(len(df1)), df1[col],label=col)
        
        ax[i].set_xlim(0,len(df1))
        if month:
            ax[i].set_title(month+str(day).zfill(2), loc="left")
        if step:
            # ax[i].set_title(month+str(day).zfill(2), loc="left")
            ax[i].set_xticks(np.arange(len(df1)))
            ax[i].set_xticklabels(df1["hhmm"].values.tolist(),rotation=0)
            st, ed = ax[i].get_xlim()
            # ax[i].xaxis.set_ticks(list_dd[0:len(df1):step])
            ax[i].xaxis.set_ticks(np.arange(st,ed,step))
        
        if i == n_last:
            """2021.06.30 sorimachi add"""
            ax[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
            # ax[i].legend(ncol=4)
            
        ax[i].set_ylim(vmin,vmax)
        ax[i].set_xlabel("時刻")
        # ax[i].set_ylabel(r"$\displaystyle Solar-Rad[W/m^2] $")
        # ax[i].set_ylabel(r"Solar-Rad[W/$m^2$]")
        ax[i].set_ylabel(r"PV[×$10^3$kWh]")
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



def plot1m_ec(df,_col,_mem_col=False,month=False,vmin=0,vmax=1000,title=False,step=6):
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
    
    if df["time"].dtypes == object:
        df["time"] = pd.to_datetime(df["time"])
    df["day"] = df["time"].apply(lambda x: x.day)
    
    last_day = np.max(df["day"])
    n_last = len(df["day"].unique())-1
    # print(n_last)
    # sys.exit()
    
    for i ,day in enumerate(_dd):
        df1 = df[df["day"]==day].reset_index(drop=True)
        # print(df1.shape)
        # continue
        # sys.exit()
        #DataFrameが0じゃなければ、描画の開始を行う予定
        if df1.shape[0] !=0:
                
            # df1.index = [ idx % int(60 / timestep) for idx in df1.index]
            # stepsize=12*6
            df1["hhmm"] = df1["time"].apply(lambda x: x.strftime("%H"))
            list_dd = df1["hhmm"].values
            
            # if figtype=="plot":
            for kk,col in enumerate(_col):
                color = _color[kk]
                ax[i].plot(np.arange(len(df1)),df1[col].values,label=col,color=color,alpha=0.8,lw=5)
                    # ax[i].plot(df1[col],label=col, color= cm.colors[kk], alpha=.7)
            
            if _mem_col:
                for k,col in enumerate(_mem_col):
                    color = "gray"
                    ax[i].plot(np.arange(len(df1)),df1[col].values,color=color,alpha=0.3,lw=1)
            
            
            ax[i].set_xlim(0,len(df1))
            if vmax:
                ax[i].set_ylim(vmin,vmax)
            if month:
                ax[i].set_title(month+str(day).zfill(2), loc="left")
                
            if step:
                # ax[i].set_title(month+str(day).zfill(2), loc="left")
                ax[i].set_xticks(np.arange(len(df1)))
                ax[i].set_xticklabels(df1["hhmm"].values.tolist())
                st, ed = ax[i].get_xlim()
                # ax[i].xaxis.set_ticks(list_dd[0:len(df1):step])
                ax[i].xaxis.set_ticks(np.arange(st,ed,step))
            
            if i == n_last:
                """2021.06.30 sorimachi add"""
                ax[i].legend(bbox_to_anchor=(1.10, 1), loc='upper left', borderaxespad=0, fontsize=18)
            # ax[i].set_ylim(vmin,vmax)
            # ax[i].set_xlabel("time[0-48]")
            # # ax[i].set_ylabel(r"$\displaystyle Solar-Rad[W/m^2] $")
            # ax[i].set_ylabel(r"Solar-Rad[W/$m^2$]")
        # sys.exit()
        else:
            # ax[i].set_visible(False)
            pass
        
        #-dd loop ----------------
    
    for i in range(n_last+1,35):
        ax[i].set_visible(False)
    
    if title:
        f.suptitle(title,fontsize=20)
    #axの間隔調整-> http://ailaby.com/subplots_adjust/
    plt.subplots_adjust(wspace=0.4, hspace=0.5)
    return plt


def plot1m_2axis(df,_col,_sub_col=False,month=False,_ylim=[0,1000,0,100],title=False,step=6):
    """
    2021.08.10
    df: "time"はdatetime
    _col: 表示するカラム
    _mem_col: mem_col
    
    """
    # check !!! --------------
    if not month:
        sys.exit("[ERROR!] please input month !")
    
    vmin_ax,vmax_ax,vmin_bx,vmax_bx = _ylim
    
    # counting !!! --------------
    #init instance
    f, ax = plt.subplots(5,7,figsize=(28,20))
    ax = ax.flatten()
    
    _t = pd.date_range(start = f"{month}010000", periods = 31, freq="1D")
    # _dd = [ t.strftime("%Y%m%d") for t in _t]
    _dd = [ t.day for t in _t]
    
    if df["time"].dtypes == object:
        df["time"] = pd.to_datetime(df["time"])
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
            df1["hhmm"] = df1["time"].apply(lambda x: x.strftime("%H"))
            list_dd = df1["hhmm"].values
            
            # if figtype=="plot":
            for kk,col in enumerate(_col):
                color = _color[kk]
                ax[i].plot(np.arange(len(df1)),df1[col].values,label=col,color=color,alpha=0.8,lw=5)
                    # ax[i].plot(df1[col],label=col, color= cm.colors[kk], alpha=.7)
            
            if _sub_col:
                bx = ax[i].twinx()
                for k,col in enumerate(_sub_col):
                    color = "gray"
                    bx.plot(np.arange(len(df1)),df1[col].values,color=color,alpha=0.7,lw=3)
                bx.set_ylim(vmin_bx,vmax_bx)
            
            
            ax[i].set_xlim(0,len(df1))
            if 1:
                ax[i].set_ylim(vmin_ax,vmax_ax)
                # ax[i].set_ylim(vmin_ax,vmax_bx)
            if month:
                ax[i].set_title(month+str(day).zfill(2), loc="left")
                
            if step:
                # ax[i].set_title(month+str(day).zfill(2), loc="left")
                ax[i].set_xticks(np.arange(len(df1)))
                ax[i].set_xticklabels(df1["hhmm"].values.tolist())
                st, ed = ax[i].get_xlim()
                # ax[i].xaxis.set_ticks(list_dd[0:len(df1):step])
                ax[i].xaxis.set_ticks(np.arange(st,ed,step))
            
            if i == n_last:
                """2021.06.30 sorimachi add"""
                ax[i].legend(bbox_to_anchor=(1.10, 1), loc='upper left', borderaxespad=0, fontsize=18)
            # ax[i].set_ylim(vmin,vmax)
            # ax[i].set_xlabel("time[0-48]")
            # # ax[i].set_ylabel(r"$\displaystyle Solar-Rad[W/m^2] $")
            # ax[i].set_ylabel(r"Solar-Rad[W/$m^2$]")
        # sys.exit()
        else:
            # ax[i].set_visible(False)
            pass
        
        #-dd loop ----------------
    
    for i in range(n_last+1,35):
        ax[i].set_visible(False)
    
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

