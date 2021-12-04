# -*- coding: utf-8 -*-
# when   : 2020.0x.xx
# who : [sori-machi]
# what : [ ]
#---------------------------------------------------------------------------
# basic-module
if 1:
  import os, sys, gc
  import glob
  import datetime as dt
  import time
  import itertools
  import importlib
  import pickle
  import warnings
  warnings.simplefilter("ignore")

  import matplotlib
  import matplotlib.ticker as mticker
  from mpl_toolkits.axes_grid1 import make_axes_locatable
  # matplotlib.use('Qt5Agg')
  import matplotlib.pyplot as plt
  # for p in sys.path:
  # sys.path.append("/home/ysorimachi/.conda/envs/sori_conda/bin")
  # sys.path.append("/opt/pyenv/versions/miniconda3-latest/envs/anaconda201910/bin")
  # print(sys.path)
  # sys.exit()
  import plotly
  import plotly.graph_objects as go

  import numpy as np
  import pandas as pd
  from pandas import DataFrame, Timestamp, Timedelta
  from pandas.tseries.offsets import Hour, Minute, Second

  try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfea
    # import cartopy.feature as cfea
    import cartopy.io.shapereader as shapereader
    from cartopy.feature import ShapelyFeature
  except:
    print("could not import cartopy")
    # https://www.naturalearthdata.com/download/10m/cultural/ne_10m_admin_1_states_provinces.zip

def load_data(path):
    df = pd.read_csv(input_path, delim_whitespace=True, header=None, names=["time","rad"])
    df["time"] = pd.to_datetime(df["time"].astype(str))
    df["rad"] = df["rad"].replace(9999,np.nan)
    df["hhmm"]  =df["time"].apply(lambda x: int(x.strftime("%H%M")))
    start_t=900
    end_t=1500
    df = df[(df["hhmm"]>=start_t)&(df["hhmm"]<=end_t)]
    df = df.drop("hhmm",axis=1)
    # df["mm"] = df["time"].apply(lambda x: x.strftime("%H%M"))
    return df

def mk_point_list(nx,ny, out_path):
    rdx  = 30./ nx
    rdy = 30./ ny
    print(rdx,rdy)
    # sys.exit()
    tbl_path = "/home/ysorimachi/work/8now_cast/out/1112_shikoku/8now0/out/20190401/201904010900_1.dat"
    names=["time","ix","iy","rad"]

    df = pd.read_csv(tbl_path, header=None, names = names, delim_whitespace=True)
    df["lon"] = df["ix"].apply(lambda x: rdx*(x-1)+120.)
    df["lat"] = df["iy"].apply(lambda x: rdy*(x)+20.)
    _name = [ "s4ku"+str(i).zfill(4) for i in range(1,len(df)+1)]
    df["name"] = _name
    df[["name","lon","lat","ix","iy"]].to_csv(out_path, index=False)
# def get_codeinfo(code):

def mk_all_concat():
    out_path="/home/ysorimachi/work/8now_cast/tbl/list_s4ku_201116.csv"
    tbl = pd.read_csv(out_path)
    # print(tbl.head())
    # sys.exit()
    datadir="/home/ysorimachi/work/8now_cast/out/1112_shikoku/8now0/out2/sites2"

    _name = tbl["name"].values.tolist()
    _file = tbl["file"].values.tolist()
    _lon = tbl["lon"].values.tolist()
    _lat = tbl["lat"].values.tolist()
    _df=[]
    for name,fname,lon,lat in zip(_name,_file,_lon,_lat):
        input_path = f"{datadir}/{fname}.dat"
        df = pd.read_csv(input_path, delim_whitespace=True, header=None,names =["time","ix","iy","rad"])
        df["rad2"] = df["rad"].rolling(6).mean()
        # ini_rad = df["rad"].values[0]
        # df.iloc[0,4] = ini_rad
        df["mm"] = df["time"].apply(lambda x: str(x)[-2:])
        df["hhmm"] = df["time"].apply(lambda x: str(x)[-4:])
        # df["rad2"].fillna(ini_rad)
        df = df.loc[(df["mm"]=="00")|(df["mm"]=="30")]
        df = df.loc[~(df["hhmm"]=="0900")].reset_index(drop=True)

        # get kaisei shisuu
        df["time"] = pd.to_datetime(df["time"].astype(str))
        df = df.set_index("time")
        df.index.name = "dti"
        df["LAT"] = lat
        df["LON"] = lon
        df["ALT"] = np.zeros(len(df))

        # df.columns = ["LAT","LON","ALT"]
        # :param dti: pandas.Timestamp Series
        # :param LAT: 緯度[°]  pandas.Series
        # :param LON: 経度[°]  pandas.Series
        # :param ALT: 高度[m]   pandas.Series
        # #get sun positinon -------------
        df = sun_position_wrapper(df, Isc=1367, intg=30/60, time_shift="C", elements={"SEL": "SEL", "I0N": "I0N"})
        df[name] = df["rad2"] / (df["I0N"] * np.sin(np.deg2rad(df["SEL"])))
        _df.append(df[name])
        if fname %100 ==0: print(fname)
    
    df = pd.concat(_df,axis=1)
    df = df.reset_index()
    df.to_csv(f"/home/ysorimachi/work/8now_cast/out/1112_shikoku/8now0/out3/all_5km_seiten.csv", index=False)
    return

def calc_cluster_csv(tmp,_n_clusters,_method,out_df,csv_path=False):
    for method in _method:
        for n_clusters in _n_clusters:
            if method=="wald":
                #--
                model = AgglomerativeClustering(n_clusters=n_clusters,linkage="ward").fit(tmp)
                out_df[f"{method}_{n_clusters}"] = model.labels_
                #------
                # df_clst, hclust = Hierarchical_Clustering(tmp_norm_df,n_clusters=n_clusters,method_name="average",path_dendrogram=f"{tden_dhome}/png/plot_dendrogram_{method}_{n_clusters}.png")
                # df_clst.to_csv(f"{tden_dhome}/cluster/{method}_df_{n_clusters}.csv", index=False)
                # # print(hclust)
                # tmp2[f"{metho}_{n_clusters}"]  = df_clst["N"]
                #------------
                # sys.exit()
            elif method=="ave":
                model = AgglomerativeClustering(n_clusters=n_clusters,linkage="average").fit(tmp)
                out_df[f"{method}_{n_clusters}"] = list(model.labels_)
            elif method=="kmeans":
                model = KMeans(n_clusters=n_clusters, random_state=0).fit(tmp)
                out_df[f"{method}_{n_clusters}"] = list(model.labels_)
            print(f"end {method} | {n_clusters}...")
    if csv_path:
        out_df.to_csv(csv_path,index=False)
    return out_df

def plot_map(df,params,png_path,isColor=False):
    
    """
    edit-day : 2021.03.15
    edit-day : 2121.06.20
    
    〇色付きカラーマップ
    input: df(pandas->DataFrame)： "lon","lat","z"(色付き)
    〇通常プロット
    input: df(pandas->DataFrame)： "lon","lat","z"があると色付きなので事前にdropしておく
    
    :params["setmap"](list) : [lon_min,lon_max, lat_min, lat_max]
    :params["cmap"]("String") : "jet"
    
    """
    # setting------------
    Grid=0
    # H,W = 10,10
    H,W = 8,8
    # isText, fontsize= True, 5
    isText, fontsize= False, 5
    
    
    # commands-----------
    _lon= list(df["lon"].values)
    _lat= list(df["lat"].values)
    
    if "z" in df.columns:
        _z= list(df["z"].values) #columns list
        isColor=True
    
  # _name = params["name"]

    lon0 = params["setmap"][0]
    lon1 = params["setmap"][1]
    lat0 = params["setmap"][2]
    lat1 = params["setmap"][3]
    
    if params["cmap"] is None:
        cmap = "jet"
    else:
        cmap = params["cmap"]
    # gspan= params["gspan"]
    #cartopy setting........

    projection=ccrs.PlateCarree()
    color_polygon = "k"
    lakes_ = cfea.NaturalEarthFeature('physical', 'lakes', '10m', edgecolor=color_polygon, facecolor="none", linewidth=0.5)
    states_ = cfea.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '10m', edgecolor=color_polygon, facecolor='none', linewidth=0.2)
    # _draw_range = [edge1["w"],edge1["e"],edge1["s"],edge1["n"]]
    # shapes = list(shapereader.Reader(input_shp_path).geometries())

    f = plt.figure(figsize=(H,W))
    # ax = f.add_subplot(1,1,1,projection=projection)
    # f,ax = plt.subplots(2,2,figsize=(2,2),projection=projection)
    # ax.add_geometries(shapes, projection, edgecolor='g', facecolor='g', alpha=0.3)
    
    ax = f.add_subplot(111,projection=projection)
    
    if Grid:        
        grd = ax.gridlines(crs=projection)
        grd.xlocator = mticker.FixedLocator(list(np.arange(lon0,lon1+0.1,1)))
        grd.ylocator = mticker.FixedLocator(list(np.arange(lat0,lat1+0.1,1)))

    ax.set_extent((lon0,lon1,lat0,lat1),projection)
    ax.coastlines(resolution="10m")
    # ax.add_feature(cfea.LAND,color='w') #land color color='#9acd32'
    # ax.add_feature(cfea.OCEAN,color='aqua') #sea olor='#7fffd4'
    # ax.add_feature(cfea.BORDERS)
    ax.add_feature(lakes_)
    ax.add_feature(states_)
    
    if isColor:
        cf = ax.scatter(_lon,_lat,c=_z, marker="s",
                        cmap=cmap,transform=projection,s=40, alpha=0.5)
        plt.colorbar(cf, pad=0.05, fraction=0.05,shrink=0.7)
    else:
        ax.scatter(_lon,_lat,marker="o",transform=projection,s=60, color="r")
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    if isText:
        for i,r in df.iterrows():
            text = r["text"]
            lon,lat = r["lon"], r["lat"]
            ax.text(lon,lat, text,fontsize=fontsize)
    # ax.colorbar(cf, cax=cax)
    ax.set_title("z", loc="left",pad=None)
    ax.margins(x=0, y=0)
        # plt.colorbar(cf, pad=0.05, fraction=0.05)
        # break
    # ax.add_feature(cfea.OCEAN,color='g') #sea olor='#7fffd4'
    plt.subplots_adjust(wspace=0.1, hspace=0.0)
    # plt.colorbar(cf, pad=0.05, fraction=0.05)
    plt.tight_layout()
    plt.savefig(png_path,bbox_inches="tight", pad_inches = 0)
    plt.close()
    return


if __name__ =="__main__" :
    #director setting........
    OUT3="/home/ysorimachi/work/8now_cast/out/1112_shikoku/8now0/out3"
    # make tbl -------------------------------------------------------
    out_path="/home/ysorimachi/work/8now_cast/tbl/list_s4ku_201116.csv"
    csv_path=f"/home/ysorimachi/work/8now_cast/out/1112_shikoku/8now0/out3/cluster_map.csv"
    # mk_point_list(nx=4800.,ny=7200, out_path=out_path)
    # sys.exit()
    df = pd.read_csv(csv_path)
    _col = [ col for col in df.columns if "wald" in col]

    _array = np.arange(2,47,4)
    # sys.exit()
    for n_sta in _array:
# lonmin, lonmax, latmin, latmax
        params={
            "setmap":tuple([131,135,32,35]),
            "lon": df["lon"].values.tolist(),
            "lat": df["lat"].values.tolist(),
            "z": [ "wald_"+str(n_sta+i) for i in range(4) ],
            # "name": df["name"].values.tolist(),
            # "vminmax":[0,n_cluster],
            "cmap": "Set1"
        }
        png_path = f"{OUT3}/png/{n_sta}.png"
        plot_map_cartopy(df,params,png_path)
        # sys.exit()
        print(f"end {n_sta}...")
        #cartopy.setting............



    sys.exit()
    #-------------------------------------------------------
    tbl = pd.read_csv(out_path)
    out_df=tbl[["name","lon","lat"]]
    # print(tbl.head())
    # sys.exit()
    # make all sites concat to 1file -------------------------------------------------------
    # mk_all_concat()
    # sys.exit()
    datadir="/home/ysorimachi/work/8now_cast/out/1112_shikoku/8now0/out2/sites2"

    input_path="/home/ysorimachi/work/8now_cast/out/1112_shikoku/8now0/out3/all_5km.csv"
    input_path="/home/ysorimachi/work/8now_cast/out/1112_shikoku/8now0/out3/all_5km_seiten.csv"
    df = pd.read_csv(input_path)
    df = df.drop(["dti"],axis=1).T.reset_index()

    tmp =df.iloc[:,1:].dropna(axis=1)
    # print(tmp.head())
    # sys.exit()


    # make all sites concat to 1file -------------------------------------------------------
    _method=["wald"]
    _n_clusters = [ i for i in range(2,51+1)]
    csv_path=f"/home/ysorimachi/work/8now_cast/out/1112_shikoku/8now0/out3/cluster_map.csv"
    calc_cluster_csv(tmp,_n_clusters,_method,out_df,csv_path=csv_path)

    sys.exit()

