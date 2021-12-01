# -*- coding: utf-8 -*-
# when   : 2020.0x.xx
# import
if 1:
  import os, sys, gc
  import glob
  import datetime as dt
  import time
  import itertools
  import importlib
  import pickle

  import matplotlib
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
    import cartopy.feature as cfeature
    import cartopy.io.shapereader as shapereader
    from cartopy.feature import ShapelyFeature
  except:
    print("could not import cartopy")


_ISC = 1367

def sun_position(doy, time, lat, lon, alt, ang=0, gam=0, Isc=_ISC, intg=1, time_shift="C"):
    """
    ※前積算時間を想定
    ※積算時間をintgで指定。サンプリングの時間粒度は任意。10分ごとのデータセットでもintg=1としてよい。
    :param doy: 年間通算日[日]  array
    :param time: 指定時刻[時]  array
    :param lat: 緯度[°]  array
    :param lon: 経度[°]  array
    :param alt: 高度[m]   array
    :param ang: 傾斜面の傾斜角[°] array ⇒　傾斜面に対する太陽光入射角の計算に必要
    :param gam: 傾斜面の方位角[°](真南0度、時計回りに0～360°) array ⇒　傾斜面に対する太陽光入射角の計算に必要
    :param time_shift:太陽位置の計算時刻を指定　
             ⇒　"C":time-intg/2(前○時間の中心時刻)で計算(夜間は中心時刻にならない)、"J":time(指定時刻)で計算
    :param Isc: 太陽定数　※最新の値は1367[W/m2]
    :param intg: 前積算時間[時間] ※前1時間ならintg=1[時間], 前10分ならintg=10/60[時間]とする
    :return:
     SEL : 計算時刻における太陽高度[°]
     SEL_noon : 正午（=地方標準時で12時）における太陽高度[°]
     AM : 計算時刻におけるエアマス
     PSD: 可照率　※時刻(time)と前積算時間(intg)における可照時間の積算時間に対する割合[時間/時間]=[無次元]
     I0N：計算日における大気外法線面日射強度[W/m2]
     SAZ : 太陽方位角[°]※真南0°,東が0～-180°, 西が0～+180°
     SIH：水平面への太陽光入射角[°]
     SIT：傾斜角angの傾斜面への太陽光入射角[°]
     sunrise : 日の出時刻[時]
     sunset : 日の入時刻[時]
     nanchu : 南中時刻[時] ※太陽が真南にいる時刻（真太陽時で12時）

    """

    deg2rad = np.pi / 180
    rad2deg = 180 / np.pi

    ## 日別太陽位置計算
    def calc_Gamma(doy):
        """
        :param doy: 年間通算日(day of year)
        :return:楕円軌道上の地球の位置（日平均） [radian]
        """
        Gamma = (2 * np.pi * (doy - 1) / 365)
        return Gamma
    def calc_delta(Gamma):
        """
        :param Gamma: 楕円軌道上の地球の位置（日平均） [radian]
        :return: 太陽赤緯(日平均) [度]
        """
        delta = (0.006918 - 0.399912 * np.cos(Gamma) + 0.070257 * np.sin(Gamma) \
                 - 0.006758 * np.cos(2 * Gamma) + 0.000907 * np.sin(2 * Gamma) \
                 - 0.002697 * np.cos(3 * Gamma) + 0.001480 * np.sin(3 * Gamma)) * rad2deg
        return delta
    def calc_Et(Gamma):
        """
        :param Gamma: 楕円軌道上の地球の位置（日平均） [radian]
        :return: 均時差[分] (＝ 真太陽時－平均太陽時)
        """
        Et = (0.000075 + 0.001868 * np.cos(Gamma) - 0.032077 * np.sin(Gamma) \
              - 0.014615 * np.cos(2 * Gamma) - 0.040849 * np.sin(2 * Gamma)) * 229.18
        return Et
    Gamma = calc_Gamma(doy)
    delta = calc_delta(Gamma)
    Et = calc_Et(Gamma)

    ## 時刻別太陽位置計算
    def calc_PSD(lat, lon, Et, delta, intg):
        """
        :param lat:緯度[°]
        :param lon: 経度[°]
        :param Et:均時差 [分]
        :param delta: 太陽赤緯[°]
        :param intg:積算時間 [時]
        :return:
          PSD：可照率　※時刻(time)と前積算時間(intg)における可照時間の積算時間に対する割合[時間/時間]=[無次元]
          sunrise：日の出時刻[時]
          sunset ：日の入時刻[時]
          nanchu ：南中時刻[時]　※太陽が真南にいる時刻（真太陽時で12時）
        """

        def Nd(lat, delta):
            """
            :param lat:緯度[°]
            :param delta: 太陽赤緯[°]
            :return: 1日の可照時間[時間]
            """
            Nd = (2 / 15) * 2 * rad2deg * np.arcsin(np.sqrt(\
                np.sin((45 + (lat - delta + 34 / 60) / 2) * deg2rad) \
                * np.sin((45 - (lat - delta - 34 / 60) / 2) * deg2rad) \
                / (np.cos(lat * deg2rad) * np.cos(delta * deg2rad)) \
                ))
            return Nd
        Nd = Nd(lat,delta)  ## 1日の可照時間[時間]
        nanchu = 12 - (4 * (lon - 135) / 60 + Et / 60)  ## 南中時刻[時]
        sunrise = nanchu - (Nd / 2)  ## 日の出時刻[時]
        sunset = nanchu + (Nd / 2)  ## 日の入時刻[時]
        PSD = np.sort(np.array([time - sunrise, np.ones(time.shape)*intg, np.ones(time.shape)*0]), axis=0)[1,:]
        ## 3つの値の中間の大きさを取得 ⇒　午前の積算時間当たりの可照時間[時]
        PSD2 = np.sort(np.array([sunset - (time - intg), np.ones(time.shape)*intg, np.ones(time.shape)*0]),axis=0)[1,:]
        ## 3つの値の中間の大きさを取得 ⇒　午後の積算時間当たりの可照時間[時]
        PSD[time>12] = PSD2[time>12]
        PSD /= intg

        return [PSD,sunrise,sunset,nanchu]
    def calc_Hs(time,lon,Et,PSD,time_shift):
        """
        :param time:指定時刻[時]
        :param lon: 経度[°]
        :param Et: 均時差[分]
        :param PSD: 可照時間[無次元]
        :param time_shift: 計算時刻を指定する識別子
        :return: 計算時刻で評価した真太陽時[時]
        """
        if time_shift == "C":
            shift = np.minimum(np.ones(PSD.shape)*1, PSD) * intg / 2  ## 夜間は正しくシフトしないが無視
            time_calc = time - shift
            time_calc2 = time - intg + shift
            time_calc[time > 12] = time_calc2[time>12]
        elif time_shift == "J":
            time_calc = time
        Hs = time_calc + 4 * (lon - 135) / 60 + Et / 60  ## 真太陽時[時]
        return Hs
    def calc_omega(Hs):
        """
        :param Hs:真太陽時[時]
        :return: 時角[°] ※南中時（太陽が真南にいる,真太陽時=12時）を0度とする、時計回りに0-360°
        """
        omega  = 15 * (Hs - 12)
        omega2 = 15 * (Hs + 12)  ## [度]
        omega[Hs < 12] = omega2[Hs < 12]

        return omega
    PSD, sunrise, sunset, nanchu = calc_PSD(lat,lon,Et,delta,intg)
    Hs = calc_Hs(time,lon,Et,PSD,time_shift)
    omega = calc_omega(Hs)
    Hs_noon = calc_Hs(np.ones(time.shape)*12,lon,Et,np.ones(PSD.shape)* 1,"J")
    omega_noon = calc_omega(Hs_noon)  ## 正午（＝地方標準時で12時）の時角

    ## 各種要素の計算
    def calc_SEL(lat, delta, omega):
        """
        :param lat:緯度[°]
        :param delta: 太陽赤緯[°]
        :param omega: 時角[°]
        :return: 太陽高度[°]
        """
        h = np.arcsin(\
            np.sin(lat * deg2rad) * np.sin(delta * deg2rad) + np.cos(lat * deg2rad) * np.cos(delta * deg2rad) \
            * np.cos(omega * deg2rad)) * rad2deg  ## [度]

        return h
    def calc_AM(h, alt):
        """
        :param h:太陽高度[°]
        :param alt: 高度[m]
        :return: エアマス
        """
        # AM = 1/np.sin(h*np.pi/180)
        AM = np.ones(h.shape)*np.nan
        AM[h>0] = ((1 - alt[h>0] / 44308) ** (5.257)) \
                  * ((np.sin(h[h>0] * deg2rad) + 0.15 * ((h[h>0] + 3.885) ** (-1.253))) ** (-1))

        return AM
    def calc_SAZ(lat,h,delta,Hs):
        """
        :param lat:緯度[°]
        :param h: 太陽高度[°]
        :param delta: 太陽赤緯[°]
        :param Hs: 真太陽時[時]
        :return: 太陽方位角[°]  ※真南0°,東が0～-180°, 西が0～+180°
        """
        omega = 15 * (Hs - 12)  ## [度]
        # psi = np.arccos((np.sin(lat*np.pi/180)*np.sin(h*np.pi/180)-np.sin(delta*np.pi/180))\
        # /(np.cos(lat*np.pi/180)*np.cos(h*np.pi/180)))*180/np.pi ## [度]
        cosA = (np.sin(lat * deg2rad) * np.sin(h * deg2rad) - np.sin(delta * deg2rad)) \
               / (np.cos(lat * deg2rad) * np.cos(h * deg2rad))  ## [度]
        sinA = np.cos(delta * deg2rad) * np.sin(omega * deg2rad) / np.cos(h * deg2rad)
        psi = np.arccos(cosA) * rad2deg  ## np.arccos : 0～180°を返す
        psi[sinA < 0] = -psi[sinA < 0]  ## 真南0°,東が0～-180°, 西が0～+180°

        return psi
    def calc_SI(lat,ang,delta,omega,gam):
        """
        :param lat:緯度[°]
        :param ang: 傾斜面の傾斜角[°]
        :param delta: 太陽赤緯[°]
        :param omega: 時角[°]
        :param gam: 傾斜面の方位角[°] ※真南を0°として時計回りに測る
        :return: 傾斜角(ang)に対する太陽光入射角[°] ※水平面では90°-h(太陽高度)と等しくなる
        ※np.arccos -> 0～180°
        """
        SI = (rad2deg) * np.arccos(\
            (np.sin(lat * deg2rad) * np.cos(ang * deg2rad) \
             - np.cos(lat * deg2rad) * np.sin(ang * deg2rad) * np.cos(gam * deg2rad)) \
            * np.sin(delta * deg2rad) \
            + (np.cos(lat * deg2rad) * np.cos(ang * deg2rad) \
               + np.sin(lat * deg2rad) * np.sin(ang * deg2rad) * np.cos(gam * deg2rad)) \
            * np.cos(delta * deg2rad) * np.cos(omega * deg2rad) \
            + np.cos(delta * deg2rad) * np.sin(ang * deg2rad) * np.sin(gam * deg2rad) * np.sin(omega * deg2rad) \
            )
        return SI
    def calc_I0N(doy, Isc):
        """
        :param doy:年間通算日[日]
        :param Isc: 太陽定数[W/m2]
        :return: 大気外法線面日射強度[W/m2]

        ↓より、E0の厳密式のソースはSpencer(1971)
        https://inis.iaea.org/collection/NCLCollectionStore/_Public/38/106/38106953.pdf
        ⇒　E0 = 1.000110+0.034221*np.cos(Gamma)+0.001280*np.sin(Gamma)+0.000719*np.cos(2*Gamma)+0.000077*np.sin(2*Gamma)

        ※NEDOでこれまで用いてきた簡易式の出所は不明
        ⇒  E0 = 1 + 0.033 * np.cos(2 * np.pi * (doy - 2) / 365)
        """

        # E0 = 1 + 0.033 * np.cos(2 * np.pi * (doy - 2) / 365)
        E0 = 1.000110+0.034221*np.cos(Gamma)+0.001280*np.sin(Gamma)+0.000719*np.cos(2*Gamma)+0.000077*np.sin(2*Gamma)
        I0 = Isc * E0

        return I0

    SEL = calc_SEL(lat, delta, omega)
    SEL_noon = calc_SEL(lat, delta, omega_noon)  ## 正午の太陽高度
    AM = calc_AM(SEL, alt)
    SAZ = calc_SAZ(lat,SEL,delta,Hs)
    SIH = calc_SI(lat, 0,delta,omega,0)
    SIT = calc_SI(lat, ang, delta, omega, gam)
    I0N = calc_I0N(doy, Isc)

    return [PSD,sunrise,sunset,nanchu,SEL,SEL_noon,AM,SAZ,SIH,SIT,I0N]
    
def sun_position_wrapper(df, Isc=_ISC, intg=1, time_shift="C", elements={"SEL":"SEL_1HC","PSD":"PSD_1H"}):
    """
    ※前積算時間を想定
    ※積算時間をintgで指定。サンプリングの時間粒度は任意。10分ごとのデータセットでもintg=1としてよい。
    df = pd.DataFrame
    df.index.name = "dti"
    df.columns = ["LAT","LON","ALT" (,"ANG","GAM")]
    :param dti: pandas.Timestamp Series
    :param LAT: 緯度[°]  pandas.Series
    :param LON: 経度[°]  pandas.Series
    :param ALT: 高度[m]   pandas.Series
    :param ANG: 傾斜面の傾斜角[°] pandas.Series
             ⇒　傾斜面に対する太陽光入射角の計算に必要
    :param GAM: 傾斜面の方位角[°](真南0度、時計回りに0～360°) pandas.Series
             ⇒　傾斜面に対する太陽光入射角の計算に必要
    :param time_shift:太陽位置の計算時刻を指定　
             ⇒　"C":time-intg/2(前○時間の中心時刻)で計算(夜間は中心時刻にならない)、"J":time(指定時刻)で計算
    :param Isc: 太陽定数　※最新の値は1367[W/m2](JMA,WMOが利用) ※年変動が0.1％程度(1366～1367[W/m2])
    :param intg: 前積算時間[時間] ※前1時間ならintg=1[時間], 前10分ならintg=10/60[時間]とする
    :param elements: dict{取得する変数名:共通要素名}  例：elements={"SEL":"SEL_10MC","I0N":"I0N_10M","PSD":"PSD_10M"}
    :return: df

    """

    # doy = (df.index - df.index.map(lambda t: t.replace(month=1, day=1))).days.values + 1  ## 年間通し日付
    dti = df.index.get_level_values(level="dti")
    doy = dti.dayofyear.values
    # time = (df.index.hour + df.index.minute / 60 + df.index.second / 3600).values  ## 単位[時]
    time = (dti.hour + dti.minute / 60 + dti.second / 3600).values  ## 単位[時]
    lat = df["LAT"].values
    lon = df["LON"].values
    alt = df["ALT"].values
    if "ANG" in df.columns:
        # print("傾斜角0°で傾斜面入射角を計算")
        ang = df["ANG"].values
    else:
        ang = np.zeros(len(df))
    if "GAM" in df.columns:
        # print("方位角0°で傾斜面入射角を計算")
        gam = df["GAM"].values
    else:
        gam = np.zeros(len(df))

    PSD,sunrise,sunset,nanchu,SEL,SEL_noon,AM,SAZ,SIH,SIT,I0N \
        = sun_position(doy,time,lat,lon,alt,ang,gam,Isc,intg,time_shift)

    for elems in elements.keys():
        col = elements[elems]
        # print(elems,col)
        df[col] = locals()[elems]

    return df

def _plotly_ts(df,path=None,n_clusters=None):
  colors = ["red", "blue", "pink", "green", "yellow", "brown", "black"]*100

  idxs = pd.date_range(start=Timestamp("2019-4-1"),end=Timestamp("2020-4-1"),freq=Minute(30))
  df = df.reindex(idxs)

  # Initialize figure
  fig = plotly.subplots.make_subplots(rows=5, cols=2, shared_xaxes="all")

  # Add Traces
  x = list(df.index)
  for i,c in enumerate(range(1,n_clusters+1)):
      for m in df.loc[:,c].columns:
          trace = go.Scatter(x=x,
                              y=list(df.loc[:,(c,m)]),
                              name="C{c:0=2}_{m}".format(c=c,m=m),
                              line=dict(color=colors[i]))
          fig.append_trace(trace, row=i%5+1, col=i//5+1)

  ## y軸のラベル
  ymin = 0
  ymax = 1
  fig.update_yaxes(title_text="CI",range=[ymin, ymax])

  plotly.offline.plot(fig, filename=path, auto_open=False)

  # fig.show()

  return None


def Hierarchical_Clustering(df,
                            n_clusters=24,
                            method_name="average",
                            path_dendrogram=None
                          ):
  """
  dat = df.values
  idx = df.index

  -> dat : n x m のnumpy array  (n:データ数, m:次元数)
  -> idx : 基データのインデックス　→　このindexに対してクラスタ情報を割り当てる

  :param n_clusters:クラスター数
  :param method_name:距離計算法（average,ward,....）
  :return:

  n_clusters = 24
  method_name = "ward"
  dat = df1.values
  idx = df1.index
  hclust.shape
  """
  from scipy import cluster

  dat = df.values
  idx = df.index

  #### 階層的クラスタリング ####
  hclust = cluster.hierarchy.linkage(dat, metric="euclidean", method=method_name)
  cluster_no = cluster.hierarchy.cut_tree(hclust, n_clusters=n_clusters)[:, 0]
  # cut = cluster.hierarchy.cut_tree(hclust, n_clusters=n_clusters)
  if path_dendrogram is not None:
      plt.figure(figsize=(20, 12))
      cluster.hierarchy.dendrogram(hclust,labels=idx)
      plt.savefig(path_dendrogram)

  #### クラスター番号とレコードの対応付け ####
  df_clst = DataFrame(np.tile(cluster_no.reshape(-1,1),2),index=idx,columns=["Corg","N"])
  N_mapper = df_clst["Corg"].value_counts().to_dict()
  df_clst.loc[:,"N"] = df_clst["N"].map(N_mapper)
  df_clst = df_clst.sort_values(by=["N","Corg"])
  df_clst["C"] = (~(df_clst["Corg"]==df_clst["Corg"].shift(1))).cumsum()

  return df_clst, hclust