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
from utils_log import log_write #(path,mes,init=False)
from getErrorValues import me,rmse,mape,r2,nrmse #(x,y)
#---------------------------------------------------
import subprocess
from tool_time import dtinc
# mms = MinMaxScaler()

from a01_99_utils import *
# from c01_som_cluster import *
from util_Data import load_rad, load_10,load_predict_Rad , load_predict_Rad2

# from util_Model2 import Resid2
import pickle


DHOME="/work2/ysorimachi/mix_som/out/syn_mesh/data"

SYNFOS_INIT = "../../src/now_jst_som.txt"
CLUSTER="/home/ysorimachi/data/synfos/som/model/cluster"
SOM_DIR="/home/ysorimachi/data/synfos/som/model/labels_synfos/train"


from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 18
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']


FIT_DIR="/home/ysorimachi/data/synfos/som/model/fitting_synfos"
ESTIMATE1="/home/ysorimachi/data/synfos/som/model/estimate1"
MODEL1="/home/ysorimachi/data/synfos/som/model/m1"
MODEL2="/home/ysorimachi/data/synfos/som/model/m2"
ERR="/home/ysorimachi/data/synfos/som/model/err"

ci_path = "/home/ysorimachi/data/synfos/som/model/err/max/ci.csv"

def get_err_machine(err_name):
    if err_name=="me":
        return me
    if err_name == "rmse":
        return rmse
    if err_name=="mape":
        return mape
    if err_name == "nrmse":
        return nrmse

def cut_time(df,st,ed):
    tmp = df[(df["hh"]>=st)&(df["hh"]<=ed)]
    return tmp
    

def get_err(df,tc,pc,err_name):
    """ 2021.09.02 """
    """ 2021.10.04 update """
    df = df.reset_index()
    # df["hh"] = df["time"].apply(lambda x: x.hour)
    print("err function 21.10.21")
    _dd = sorted(df["dd"].values.tolist())
    _e =[]
    for dd in _dd:
        tmp = df[df["dd"]==dd]
        tmp["diff"] = tmp[pc] - tmp[tc]
        tmp["dif2"] = np.abs(tmp[pc] - tmp[tc])
        tmp = tmp.sort_values("dif2",ascending=False)
        e = tmp["diff"].values[0]
        _e.append(e)
    
    max0=np.max(_e)
    min0=np.min(_e)
    ave = np.mean(_e)
    std = np.std(_e)
    
    return max0,min0,ave,std

def calc_mm_err(df, err_path):
    
    _mm = list(range(1,12+1))
    err_hash = {}
    _col = ['MIX','PRED1','PRED2']
    _err = ["me","rmse","mape","nrmse"]
    
    _err_col = [ f"{c}_{ename}" for ename in _err for c in _col]
    # print(_err_col)
    # sys.exit()
    for mm in _mm:
        tmp = df[df["mm"]==mm]
        
        if tmp.shape[0]>50:
            _err_v = []
            for err_col in _err_col:
                pc,err_name = err_col.split("_")
                e = get_err(tmp,"OBS",pc,err_name)
                _err_v.append(e)
        
        else:
            _err_v = [9999. for _ in range(len(_err_col))]
        
        err_hash[mm] = _err_v
    
    df = pd.DataFrame(err_hash).T
    df.index.name = "mm"
    df.columns = _err_col
    
    df.to_csv(err_path)
    return

def calc_cls_err(df, err_path):
    
    # _mm = list(range(1,12+1))
    err_hash = {}
    _col = ['MIX','PRED1','PRED2']
    _err = ["me","rmse","mape","nrmse"]
    
    _err_col = [ f"{c}_{ename}" for ename in _err for c in _col]
    for c in _col:
        df[c] = df[c] - df["OBS"]
    
    df = cut_time(df,9,14)
    # print(df.head())
    # sys.exit()
    
    #----------------------------------------------------
    if 0: #全体のboxplotの表示
        _col2 = ["CLUSTER"] + _col
        df = df[_col2]
        f,ax = plt.subplots(figsize=(18,8))
        df = pd.melt(df,id_vars=["CLUSTER"],value_vars = _col,var_name="METHOD",value_name="DIFF")


        sns.boxplot(x="CLUSTER",y ="DIFF",hue="METHOD", data=df, ax=ax)
        ax.set_ylim(-1000,1000)
        ax.set_ylabel(r"Diff Rad[W/$m^2$] Box-Plot per cluster")
        f.savefig(f"{ERR}/png/cls2/box_plot.png",bbox_inches="tight")
        plt.close()
        sys.exit()
    #----------------------------------------------------
    # print(_err_col)
    # sys.exit()
    _count = []
    f,ax = plt.subplots(4,4,figsize=(18,8))
    ax = ax.flatten()
    cut_bins =[-1000,-200,-100,-50,0,50,100,200,1000]
    # print(df.head())
    # sys.exit()
    # f,ax = plt.sunplots(4,4,figsize=(18,8))
    _col = ['MIX','PRED1']
    for lbl in range(16):
        i = lbl
        tmp = df[df["CLUSTER"]==lbl]
        # print(tmp.shape)
        if tmp.shape[0]>0:
            _err_v = []
            _count.append(tmp["dd"].nunique())
            _df=[]
            for c in _col:
                tmp[c] = pd.cut(tmp[c],cut_bins)
                t2 = tmp.groupby(c).count()
                _df.append(t2["time"])
            
            
            tmp = pd.concat(_df,axis=1)
            tmp.columns = _col
            w = 1/(len(tmp.columns)+1)
            for j,c in enumerate(tmp.columns):
                ax[i].bar(0.5*w +np.arange(len(tmp))+w*j,tmp[c],width=w,label=c)
                ax[i].set_title(f"CLS{str(lbl).zfill(2)}")
                ax[i].set_xticks(np.arange(len(tmp)))
                ax[i].set_xlim(0,len(tmp))
                # print(tmp.index)
                # sys.exit()
                ax[i].set_xticklabels(list(tmp.index),rotation=80, fontsize=8)
                
        else:
            ax[i].set_visible(False)
    
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    f.savefig(f"{ERR}/png/cls2/cut_plot.png",bbox_inches="tight")
    # df = pd.DataFrame(err_hash).T
    # df.index.name = "cls"
    # df.columns = _err_col
    # df["count"] = _count
    # df.to_csv(err_path)
    return




def err_make1(cele="MSPP",ecode="ecmf001",resid2_name="lgb"):
    """
    concat 
    2021.10.12 stat !!
    """
    # FIT_DIR="/home/ysorimachi/data/synfos/som/model/fitting_synfos"
    
    def cut_time(df,st,ed):
        tmp = df[(df["hh_int"]>=st)&(df["hh_int"]<=ed)]
        return tmp
    
    def clensing(df,subset_col):
        # subset_col = x_col + [y_col]
        for c in subset_col:
            df[c] = df[c].apply(lambda x: isFloat(x))
        df = df.dropna(subset=subset_col)
        return df
    
    #----------------------------
    # <--- load rad ---> 
    df = load_predict_Rad(cele,ecode,resid2_name)
    df0 = df[df["istrain"]==1]
    df1 = df[df["istrain"]==0]
    #----------------------------
    _name = ["train","test"]
    for i,df in enumerate([df0,df1]):
        name = _name[i]
        # train/ test
        #----------月別----
        # err_path = f"{ERR}/mm/{name}_{resid2_name}_{ecode}_{cele}.csv"
        # calc_mm_err(df,err_path)
        # sys.exit()
        #----------クラスタ別----
        err_path = f"{ERR}/cls/{name}_{resid2_name}_{ecode}_{cele}.csv"
        calc_cls_err(df,err_path)
        # sys.exit()
    return 
    

def err_make2(cele="MSPP",ecode="ecmf001",resid2_name="lgb"):
    """
    concat 
    2021.10.20 stat !!
    2021.11.22 stat !!
    """
    # FIT_DIR="/home/ysorimachi/data/synfos/som/model/fitting_synfos"
    
    def cut_time(df,st,ed):
        tmp = df[(df["hh"]>=st)&(df["hh"]<=ed)]
        return tmp
    
    def clensing(df,subset_col):
        # subset_col = x_col + [y_col]
        for c in subset_col:
            df[c] = df[c].apply(lambda x: isFloat(x))
        df = df.dropna(subset=subset_col)
        return df
    
    def mix_day_err(df,_col):
        _dd = sorted(df["dd"].unique().tolist())
        
        err_hash = {}
        for dd in tqdm(_dd):
            tmp = df[df["dd"]==dd]
            _e=[]
            for c in _col:
                
                tmp[f"diff_{c}"] = tmp[c] - tmp["OBS"]
                tmp[f"diff2_{c}"] = np.abs(tmp[c] - tmp["OBS"])
                tmp = tmp.sort_values(f"diff2_{c}", ascending=False)
                e = tmp[f"diff_{c}"].values[0]
                _e.append(e)
            err_hash[dd] = _e
        
        df = pd.DataFrame(err_hash).T
        df.index.name = "dd"
        df.columns = _col
        df = df.reset_index()
        return df
    #----------------------------
    # <--- load rad ---> 
    df = load_predict_Rad(cele,ecode,resid2_name,n_dim=3,isCNN=1)
    df2 = load_predict_Rad2(cele,ecode,resid2_name,n_dim=3,isCNN=1) #2021.11.22
    df = df.merge(df2, on="time", how="inner")
    # print(df.head())
    df = df.rename(columns= {f"PRED_{resid2_name}": "PRED3"})
    _col = ['MIX','PRED1','PRED2',"PRED3"]
    # use_col = ['time', 'OBS',"istrain","mm","dd","hh"] + ['MIX','PRED1','PRED2',f"PRED_{resid2_name}"]
    # df = df[use_col]
    # sys.exit()
    # df0 = df[df["istrain"]==0]
    # df1 = df[df["istrain"]==1]
    #---------------------------
    
    df = cut_time(df,9,14)
    res = mix_day_err(df,_col)
    
    res = res.sort_values("PRED1")
    res = train_flg2(res)
    res.to_csv(f"{ERR}/max/day_max_err_{ecode}_{cele}_{resid2_name}.csv",index=False)
    return 


def select_day(cate,cele,ecode,resid2_name):
    """
    上振れ改善の上位30事例/下振れ改善上位事例
    """
    path = f"{ERR}/max/day_max_err_{ecode}_{cele}_{resid2_name}.csv"
    df = pd.read_csv(path)
    if cate=="U":
        df = df[df["PRED1"]>0]
    else:
        df = df[df["PRED1"]<0]
    
    df["r_pred1"] = kaizen_rate(df,"MIX","PRED1")
    df = df.sort_values("r_pred1",ascending=False)
    return df

RAD="/home/ysorimachi/data/synfos/som/model/err/png/max"
def plotRad35(cate,cele,ecode,resid2_name):
    from plot_multi_rad import plot_multi_rad #(df,_dd,_col,_sub_col=None,vmin=0,vmax=1000, return_ax=False)
    df = select_day(cate,cele,ecode,resid2_name)
    _dd = df["dd"].astype(str).values[:35]
    
    #----------dataset
    df = load_predict_Rad(cele,ecode,resid2_name)
    # _df = [ df[df["dd"]==str(dd)] for dd in _dd ] #選択
    
    
    _col = ["OBS","MIX","PRED1","PRED2"]
    _sub_col=["CR0"]
    f = plot_multi_rad(df,_dd,_col,_sub_col=_sub_col,vmin=0,vmax=1200, return_ax=False)
    f.savefig(f"{RAD}/rad35/{cate}_{cele}_{ecode}_{resid2_name}.png", bbox_inches="tight")
    plt.close()
    return

def plotRad1(cate,cele,ecode,resid2_name):
    from plot_multi_rad import plot_multi_rad #(df,_dd,_col,_sub_col=None,vmin=0,vmax=1000, return_ax=False)
    df = select_day(cate,cele,ecode,resid2_name)
    _dd = df["dd"].astype(str).values[:5]
    df = df.set_index("dd")
    
    def plt_ts_dd(ax,df,out_png):
        _main = ['OBS','MIX','PRED1',"PRED2"]
        _sub = ["SYN","EC","CR0"]
        df = df.set_index("time")
        for c in _main:
            ax.plot(np.arange(0,len(df)),df[c],label = c, lw=5)
        for c in _sub:
            ax.plot(np.arange(0,len(df)),df[c],label = c, lw=1)
        
        ax.legend(loc= "upper left")
        ax.set_ylim(-100,1200)
        ax.set_xlabel("hh")
        ax.set_ylabel(r"solar Rad [W/$m^2$]")
        ax.set_xlim(0,len(df))
        _index = [ time.strftime("%H:%M") for time in  df.index ]
        
        #区間色塗り
        ax.axvspan(0, 4, color="gray", alpha=0.5)
        ax.axvspan(20, len(df), color="gray", alpha=0.5)
        
        
        ax.set_xticks(np.arange(0,len(df)))
        ax.set_xlim(0,len(df))
        ax.set_xticklabels(_index, fontsize=12,rotation=45)
        
        return ax
    
    rad = load_predict_Rad(cele,ecode,resid2_name)
    ruond2 = lambda x : np.round(x,2)
    for i ,(dd,r) in enumerate(list(df.iterrows())[:10]):
        
        ctop_num=str(i+1).zfill(5)
        tmp = rad[rad["dd"] == str(dd)]  
        e_mix,e_p0,e_p1 = map(ruond2, [ r["MIX"],r["PRED1"],r["PRED2"] ] )

        rate0 = np.round( -1 * (e_p0 - e_mix)*100/e_mix,2)
        rate1 = np.round( -1 * (e_p1 - e_mix)*100/e_mix,2)
        

        out_png = f"{RAD}/rad1/{cate}_{ctop_num}_{cele}_{ecode}_{resid2_name}.png"
        f,ax = plt.subplots(figsize=(15,8))
        ax = plt_ts_dd(ax,tmp,out_png)
        ax.set_title(f"DAY[{dd}] - ecode003(TOKYO)-{ctop_num}_{cele}_{resid2_name}")
        
        ax.text(6,1100, rf"MAX-DIFF[W/$m^2$] PRED0({e_mix})->({e_p0}) Improve: {rate0}[%]")
        ax.text(6,1030, rf"MAX-DIFF[W/$m^2$] PRED1({e_mix})->({e_p1}) Improve: {rate1}[%]")
        f.savefig(out_png, bbox_inches="tight")
        plt.close()
        print(datetime.now(), "[END]" , i, f"{RAD}/rad1")
        # sys.exit()
    return

def calc_ci_day_list():
    """[summary]
    事前に日別のCIのリストを作成して今後に活用する
    """
    #晴天度合いの算出 ---------------
    df2 = load_predict_Rad(cele="MSPP",ecode="ecmf003",resid2_name="lgb")
    _dd = sorted(df2["dd"].unique().tolist())
    #-------------
    if 1:
        calc_ci_day(cut_time(df2,10,14),ci_path)
    return 
    # -------------------------------


def max_err_distribution(cele,ecode,resid2_name):
    """ 上振れ改善の上位30事例/下振れ改善上位事例 """
    path = f"{ERR}/max/day_max_err_{ecode}_{cele}_{resid2_name}.csv"
    df = pd.read_csv(path)
    df["dd"] = df["dd"].astype(str)
    # cut_bins =[-1000,-200,-100,-50,0,50,100,200,1000]
    cut_bins =[-1000,-500,-300,-150,0,150,300,500,1000]
    
    # print(df.head())
    # sys.exit()
    
    #晴天度合いの算出 ---------------
    if 0:
        calc_ci_day_list()
        sys.exit()
    # -------------------------------
    
    rad = pd.read_csv(ci_path)

    rad["dd"] = rad["dd"].astype(str)
    rad["ci_g"] = pd.cut(rad["mean"],[0,0.35,0.90,10])
    
    #---------------CI-valueによるクラスタされた日の抽出----
    dd_hash = {}
    for v in rad["ci_g"].unique():
        tmp = rad[rad["ci_g"]==v]
        
        _dd = tmp["dd"].astype(str).values.tolist()
        dd_hash[v] = _dd
    #------------------------
    
    
    df = df.merge(rad, on="dd", how="left")
    df = train_flg2(df) # dd-> flg
    # print(df.shape)
    df = df[df["istrain"]==0]
    # print(df.shape)
    # sys.exit()

    # for v in 
    # print(len(df2["dd"].values.tolist()))
    f,ax = plt.subplots(3,1,figsize=(18,15))
    
    # print(sorted(list(dd_hash.keys())))
    # sys.exit()
    
    for j,v in enumerate(sorted(list(dd_hash.keys()))):
        _dd = dd_hash[v]
        df2 = df.loc[df["dd"].isin(_dd),:]
        # print(df2.head())
        # sys.exit()
        # print(len(_dd))
        # sys.exit()
        # CI条件付きの分布作成 ---------------------
    
        # _col = ["MIX","PRED1","PRED2"]
        # _col2 = ["現行","統合1",f"統合2({resid2_name})"]
        _col = ["MIX","PRED1","PRED2","PRED3"]
        _col2 = ["現行","統合1",f"統合2({resid2_name})",f"統合3({resid2_name})"]
        _t=[]
        for c in _col:
            # print(df2.shape)
            # sys.exit()
            df2["bin"] = pd.cut(df2[c], cut_bins)
            t = df2.groupby("bin").count()["MIX"]
            
            # print(t.head())
            # sys.exit()
            _t.append(t)
        
        df3 = pd.concat(_t,axis=1)
        # print(df.head())
        # sys.exit()
        df3.columns = _col2
        df3["range"] = str(v)

        df3.to_csv(f"{ERR}/max/dist_CI_group{j}_{ecode}_{cele}_{resid2_name}.csv",encoding="shift-jis")
        print(f"{ERR}/max/dist_CI_group{j}_{ecode}_{cele}_{resid2_name}.csv")
    return

def plot_dist_ci(cele,ecode,resid2_name):
    f,ax = plt.subplots(3,1,figsize=(18,15),sharex=True)
    DIR="/home/ysorimachi/data/synfos/som/model/err/max"
    
    sum0 =0
    for i in range(3):
        path = f"{DIR}/dist_CI_group{i}_{ecode}_{cele}_{resid2_name}.csv"
        df = pd.read_csv(path,encoding="shift-jis")
        sum0 += np.sum(df["現行"])
    #--------------------
    for i in range(3):
        path = f"{DIR}/dist_CI_group{i}_{ecode}_{cele}_{resid2_name}.csv"
        df = pd.read_csv(path,encoding="shift-jis")

        crange = df["range"].values[0].replace("(","").replace("]","").replace(", ","-")
        df = df.set_index("bin")
        N= np.sum(df["現行"])
        p= np.round(N*100/sum0,1)
        ax[i] = sub_plot_bar(ax[i],df)
        title = f"{crange} (N={N} | {p}%)"
        ax[i].set_title(title)

    f.savefig(f"{DIR}/dist_group.png", bbox_inches="tight")
    plt.close()
    print(f"{DIR}/dist_group.png")
    return    

def get_CI_days(ci_range=[0.65,None]):
    c0,c1 = ci_range
    ci_path = "/home/ysorimachi/data/synfos/som/model/err/max/ci.csv"
    # calc_ci_day(cut_time(rad,10,14),ci_path)
    rad = pd.read_csv(ci_path)
    rad["dd"] = rad["dd"].astype(str)
    # rad["ci_g"] = pd.cut(rad["mean"],[0,0.15,0.65,10])
    if c1 == None:
        rad = rad[(rad["mean"]>c0)]
    else:
        rad = rad[(rad["mean"]>c0)&(rad["mean"]<c1)]
    print(rad.shape)
    print(rad.head())
    sys.exit()    


if __name__== "__main__":
    #----------------------------------------
    log_path = "./log_err.log"
    log_write(log_path,"start! ",init=True)
    # _cele = ["MSPP","LOCA","MICA","HICA"]
    _ecode = ["ecmf003"] #東京のみ
    # _cele = ["MICA","HICA"]
    _cele = ["HICA"]
    # _resid2 = ["lasso","ridge","svr","tree","rf","lgb"] + ["mlp3","mlp5"]
    _resid2 = ["ridge","svr","tree"]
    #------------------------------
    #-------------------------------------------------
    if 0:
        for resid2_name in _resid2:
            for ecode in _ecode:
                for cele in _cele:
                    if 1: #日別の誤差を表示するようなデータセットを作成する ---
                        err_make2(cele=cele,ecode=ecode,resid2_name=resid2_name)
                        # sys.exit()
                    # for cate in ["L"]: # U/L
                    #     # plotRad35(cate,cele,ecode,resid2_name)
                    #     plotRad1(cate,cele,ecode,resid2_name)
                    #     sys.exit()
                    # sys.exit()
                    
                    
                    print(datetime.now(),"[END]",ecode,cele,resid2_name)
                    log_write(log_path,f"[END] {ecode} {cele} name={resid2_name}",init=False)
                    # sys.exit()
                    
    if 1:
        for resid2_name in _resid2:
            for ecode in _ecode:
                for cele in _cele:
                    max_err_distribution(cele,ecode,resid2_name)
                    plot_dist_ci(cele,ecode,resid2_name)
                    sys.exit()
                
    
