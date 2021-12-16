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
# import plotly
#---------------------------------------------------
# sori -module
sys.path.append('/home/ysorimachi/tool')
# from getErrorValues import me,rmse,mae,r2 #(x,y)
from convSokuhouData import conv_sfc #(df, ave=minutes,hour)
# from mapbox import map_lonlat #(df,text="name",size=4,zoom=4)
# #amedas relate 2020.02,04 making...
# from tool_AMeDaS import code2name, name2code
# from tool_110570 import get_110570,open_110570
# from tool_100571 import get_100571,open_100571
#(code,ini_j,out_d)/(code,path,csv_path)
from sklearn.preprocessing import StandardScaler
#---------------------------------------------------
import subprocess

try:
  from a01_99_utils import *
except:
  from som_data.a01_99_utils import *



DAT="/work2/ysorimachi/synfos/data/SYNFOS-solar_ver2/maeda0421"
WORKSPACE="/work2/ysorimachi/synfos/data/SYNFOS-solar_ver2/maeda0421/dataset" # fct
WORKSPACE2="/work2/ysorimachi/synfos/data/SYNFOS-solar_ver2/maeda0421/dataset2" # obs -fct
WORKSPACE3="/work2/ysorimachi/synfos/data/SYNFOS-solar_ver2/maeda0421/dataset3" # obs -fct
DATASET_210625="/work2/ysorimachi/synfos/data/SYNFOS-solar_ver2/maeda0421/dataset_210625" # obs -fct

TBL_PATh="/home/ysorimachi/work/ecmwf/tbl/list_10.tbl"

TMP="/home/ysorimachi/work/synfos/tmp/maeda0421"
LOGFILE = "../log/mk_dataset.log"

# setting(date)-------------------------------
def load_month2(yy):
  """[summary] year入力で、月と日を返す　2021は逐次更新
  Args:
      yy ([type]): [description]

  Returns:
      [type]: [description]
  """
  if yy==2021:
    _month = ["202104","202105","202106","202107","202108","202109"]
    _day = [30,31,30,31,31,30]
  if yy==2020:
    _month = ["202004","202005","202006","202007","202008","202009","202010","202011","202012","202101","202102","202103"]
    _day = [30,31,30,31,31,30,31,30,31,31,28,31]
  if yy==2019:
    _month = ["201904","201905","201906","201907","201908","201909","201910","201911","201912","202001","202002","202003"]
    _day = [30,31,30,31,31,30,31,30,31,31,29,31]
  if yy==2018:
    _month = ["201804","201805","201806","201807","201808","201809","201810","201811","201812","201901","201902","201903"]
    _day = [30,31,30,31,31,30,31,30,31,31,28,31]
  return _month,_day


def load_mm():
  return ["04","05","06","07","08","09","10","11","12","01","02","03"]



def load_hhj():
  return [0,3,6,9,12,15,18,21]

# setting(point)-------------------------------
def load_10():
  # TBL_PATh="/home/ysorimachi/work/ecmwf/tbl/list_10.tbl"
  tbl = pd.read_csv(TBL_PATh, header=None,delim_whitespace=True)
  ec_code = tbl[0].values.tolist()
  ame_code = tbl[19].values.tolist()
  names = tbl[27].values.tolist()
  return ec_code, ame_code, names

#data (obs)
def mk_obs(month,acode):
  """"[summary]"
  init: 2021.05.03
  update: 2021.11.18
  Args:
      month ([type]): [description]
      acode ([type]): [description] 47xxx code

  Returns:
      [type]: [description]
  """
  #local function
  def clensing(df):
    df["obs"] = df["obs"].apply(lambda x: np.nan if x>1400 or x<0 else x)
    return df
  #--------------------
  # AME_HOME="/work2/ysorimachi/ec/ame"
  if int(month)<=202003 or int(month)>=202010:
    AME="/work/ysorimachi/era5/dat4/sfc0" #2021.11.17 update
  else:
    # AME_HOME="/work2/ysorimachi/ec/ame"
    AME=f"/work2/ysorimachi/ec/ame/data2" #2021.5.03 init
  path = f"{AME}/sfc_10minh_{month}_{acode}.csv"
  df = pd.read_csv(path)
  df = conv_sfc(df,ave=False)
  df = df.rename(columns={"tenminSunshine":"obs"})
  df["obs"] = df["obs"].rolling(3).mean().apply(lambda x: np.round(x,1))
  use_col =['time','tenminPrecip','obs', 'sixtyminSunshine', 'snowDepth','stationPressure', 'seaLevelPressure', 'humidity']
  df = df[use_col]
  df = clensing(df)
  #average 2021.06.25
  df["obs"] = df["obs"].rolling(3).mean()
  df = df.iloc[2::3,:].reset_index(drop=True)
  return df

#-----------------------------------------
# make_fct(ecode,month,day,czz,fd,mtd)
def mk_fct(code,month,day,czz,fd,mtd=1):
  #---- local function ----
  def syn_utc(x,fd):
    x = x - timedelta(days=fd)
    x = x - timedelta(hours=9)
    return x
  #----------------------
  _ini_j = pd.date_range(start=f"{month}01{czz}00", periods=day,freq="D") #検証で利用日配列
  _ini_fu = [ syn_utc(day,fd).strftime("%Y%m%d%H%M") for day in _ini_j]
  
  _df = []
  for ini_j,ini_u in zip(_ini_j,_ini_fu):
    path = f"{DAT}/{mtd}/{ini_u}/{code}.dat"
    df = pd.read_csv(path, delim_whitespace=True,header=None,names=["time","mix","syn","ecm","ecc","fecm","fecc"])
    
    dd = ini_j.strftime("%Y%m%d") #selectday
    df["time"] = pd.to_datetime(df["time"].astype(str))
    df["dd"] = df["time"].apply(lambda x: x.strftime("%Y%m%d"))
    df=df[df["dd"]==dd]
    _df.append(df)
  df = pd.concat(_df,axis=0)
  # df.to_csv(f"{WORKSPACE}/{code}_{month}_{czz}Z_{fd}_{mtd}.csv",index=False) #max 3480 csvs
  return df


def mk_dataset():
  """
  2021.05.06 calc start...datasetを作成する。
  """
  #init setting
  _ec_code, _ame_code, _names = load_10()
  _yy = [2019,2020]
  _hhj=[0,3,6,9,12,15,18,21]
  #------------------------------
  
  for yy in _yy:
    _month,_day = load_month2(yy)
    for month,day in zip(_month,_day):
      for ecode, acode,name in zip(_ec_code, _ame_code, _names):
        for hhj in _hhj:
          czz = str(hhj).zfill(2)
          for fd in [1,2]:
            for mtd in [1,2]:
              
              # main-----------------
              try:
                obs = mk_obs(month,acode)
                fct = mk_fct(ecode,month,day,czz,fd,mtd=mtd)
                data = pd.merge(fct,obs, on="time", how="inner")
                flg=1
                data.to_csv(f"{WORKSPACE2}/{ecode}_{month}_{czz}Z_{fd}_{mtd}.csv")
                #bkup(obs-fct dataset)
              except:
                flg=0
              # log------------------
              with open(LOGFILE, "a") as f:
                now = datetime.now()
                text = f"{now} {ecode} {month} {czz}Z {fd} {mtd} {flg}\n"
                f.write(text)
                
                
def point_on_map():
  tbl = pd.read_csv(TBL_PATh, header=None,delim_whitespace=True)
  tbl = tbl[[0,1,2,27]]
  
  tbl.columns = ["point","lon","lat","name"]
  tbl["name2"] = tbl[["name","point"]].apply(lambda x: x[0]+'('+x[1]+')',axis=1) 
  fig = map_lonlat(tbl,text="name2",size=8,zoom=4)
  html_path = f"{TMP}/map/point_map.html"
  plotly.offline.plot(fig, filename=html_path)  # ファイル
  


def get_2yy_dataset(ecode,fd=1,hhj=12):
  
  """
  month　別解析
  2021.05.10 
  """
  _hhj = load_hhj()
  _mm = load_mm()
  
  def hhj2hhZ(hhj):
    if hhj==0:
      return "15Z"
    elif hhj==3:
      return "18Z"
    elif hhj==6:
      return "21Z"
    elif hhj==9:
      return "00Z"
    elif hhj==12:
      return "03Z"
    elif hhj==15:
      return "06Z"
    elif hhj==18:
      return "09Z"
    elif hhj==21:
      return "12Z"
    else:
      sys.exit("Error! please HHJ")

  def calc_err(err_name,xcl,_ycl, data):
    _err=[]
    if err_name=="rmse":
      for ycl in _ycl:
        _err.append(rmse(data[xcl], data[ycl]))
    if err_name=="me":
      for ycl in _ycl:
        _err.append(me(data[xcl], data[ycl]))
    return _err
  
  def clensing(df , _ycl):
    for ycl in _ycl:
      df[ycl]  = df[ycl].apply(lambda x: np.nan if x>1400 else x)
    df = df.dropna()
    return df
  
  # f,ax= plt.subplots(len(_mm),len())
  err_hash = {}
  _df_all=[]
  for j,hhj in enumerate(_hhj):
    czz = str(hhj).zfill(2)
  zz = hhj2hhZ(hhj)
  # 2 years data conat ---------------------
  _df = [] 
  for yy in [2019,2020]:
    path = f"{WORKSPACE3}/{ecode}_{czz}Z_{yy}_FD{fd}.csv"
    _df.append(pd.read_csv(path))
  df = pd.concat(_df,axis=0)
  df = clensing(df,["syn","ecm1","ecm2"])
  
  df["time"] = pd.to_datetime(df["time"])
  df["month"] = df["time"].apply(lambda x: x.strftime("%m"))
  df["day"] = df["time"].apply(lambda x: x.strftime("%Y%m%d"))
  df["zz"] = zz
  # 2 years data conat --------------------
  return df

def hhj2hhZ(hhj):
  if hhj==0:
    return "15Z"
  elif hhj==3:
    return "18Z"
  elif hhj==6:
    return "21Z"
  elif hhj==9:
    return "00Z"
  elif hhj==12:
    return "03Z"
  elif hhj==15:
    return "06Z"
  elif hhj==18:
    return "09Z"
  elif hhj==21:
    return "12Z"
  else:
    sys.exit("Error! please HHJ")


def mk_rad_210625(ecode="ecmf003",fd=1):
  """
  2021.06.25 
  12Z/統合予測の出力に快晴時全天日射量/大気外全天日射量を表示するプログラムの表示
  """
  # DHOME="/work2/ysorimachi/synfos/data/SYNFOS-solar_ver2/maeda0624/2"
  DHOME="/work2/ysorimachi/synfos/data/SYNFOS-solar_ver2/maeda0421/2" #3年分ファイル
  DHOME2="/work2/ysorimachi/synfos/data/SYNFOS-solar_ver2/maeda0624/2"
  _dd = sorted(os.listdir(DHOME))
  _dd = [ t for t in _dd if t[8:10]=="12" ]
  # print(len(_dd))
  # sys.exit()
  
  #localc function ......
  def clensing(df):
    df["time"] = pd.to_datetime(df["time"].astype(str))
    for c in ["mix2","syn","ecm2"]:
      df[c] = df[c].apply(lambda x: np.nan if x==9999 or x<0 or x>1467 else x)
    return df
  
  def point_hash(ecode="ecmf003",cate="scode"):
    _ecode,_scode,_name= load_10()
    point_codes = { k:v for k,v in zip(_ecode, _scode)}
    point_names = { k:v for k,v in zip(_ecode, _name)}
    if cate =="scode":
      return point_codes[ecode]
    else:
      return point_names[ecode]
    
  def mk_obs_all():
    scode = point_hash(ecode,cate="scode")
    _df = []
    # for yy in [2019,2020]:
    for yy in [2018,2019,2020,2021]:
      _month,_day = load_month2(yy)
      for month in _month:
        _df.append(mk_obs(month,scode))
    df = pd.concat(_df,axis=0)
    return df
  
  #------OBS ALL year data -----  "
  print(datetime.now(),"[start]", "make OBS->", ecode)
  obs = mk_obs_all()[["time","obs"]]
  #------OBS ALL year data -----  "s
  #localc function ......
  names = ["time","mix2","syn","ecm2","ecm2c","fecm","fecc","rCR0","rS0"]
  _df = []
  for ini_u in tqdm(_dd):
    # ini_u="202104021200"
    if int(ini_u[:6])>=201904 and int(ini_u[:6])<=202103:
      path = f"{DHOME2}/{ini_u}/{ecode}.dat" #CR0もあるdata
    else:
      path = f"{DHOME}/{ini_u}/{ecode}.dat"
      
    df = pd.read_csv(path,header=None,delim_whitespace=True,names=names)
    # print(df.head())
    df = clensing(df)
    # 12Z/統合予測の出力に快晴時全天日射量/大気外全天日射量を表示するプログラムの表示翌日予測
    if fd==0: #当日予測
      df = df.iloc[7:7+48,:]
    if fd==1: #翌日予測
      df = df.iloc[55:55+48,:]
    if fd==2:
      df = df.iloc[103:103+48,:]
    
    _df.append(df)
  
  fct = pd.concat(_df,axis=0)
  df = fct.merge(obs,on="time",how="inner")
  df.to_csv(f"{DATASET_210625}/{ecode}_FD{fd}.csv", index=False)
  print(datetime.now(),"[end]",ecode,"FORECAST time -> ", fd)
  return

def load_rad(ecode="ecmf003",fd=1):
  """
  2021.06.25 東京地点の2年分の日射量データセット作成済/以降はここから読込
  2021.07.20 ecmwfの解析用にここからデータ取得
  """
  path = f"{DATASET_210625}/{ecode}_FD{fd}.csv"
  if not os.path.exists(path):
    mk_rad_210625(ecode=ecode,fd=fd)
  df = pd.read_csv(path)
  df["time"] = pd.to_datetime(df["time"])
  df["month"] = df["time"].apply(lambda x: x.strftime("%Y%m"))
  df["day"] = df["time"].apply(lambda x: x.strftime("%Y%m%d"))
  df["mm"] = df["time"].apply(lambda x: x.strftime("%m"))
  df["hh_int"] = df["time"].apply(lambda x: int(x.strftime("%H%M")))
  
  return df


def load_weather_fcs(ecode="ecmf001",ini_j="20190401",normalize=True):
  
  
  OHOME="/work2/ysorimachi/mix_som/out/syn_data/data"
  list_d = sorted(os.listdir(OHOME)) #20180401 ~ 20211102
  # print(list_d[0], list_d[-1])
  # sys.exit()
  DIR=f"{OHOME}/{ini_j}"
  # _cate = ["70RH","70UU","70VV","85RH","HICA","LOCA","MICA"]
  _cate = ["70RH","85RH","70UU","70VV","LOCA","MICA"]
  # _cate = np.unique([ os.path.basename(f).split("_")[0] for f in glob.glob("/work2/ysorimachi/mix_som/out/syn_data/data/20191010/*_mean.csv")])
  _cate = ['30RH','50RH','70OO','70RH','70UU','70VV','85OO','85RH','85UU','85VV',
 'HICA','LOCA','MICA','MSPP']
  # print(_cate)
  # sys.exit()
  
  _df = []
  for cate in _cate:
    path = f"{DIR}/{cate}_mean.csv"
    names = ['time', 'ecmf001', 'ecmf002', 'ecmf003', 'ecmf004', 'ecmf005',
       'ecmf006', 'ecmf007', 'ecmf008', 'ecmf009', 'ecmf010']
    df = pd.read_csv(path)
    df.columns = names
    # df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    _df.append(df[ecode])
  
  df = pd.concat(_df,axis=1)
  df.columns = _cate
  
  if normalize:
    ssd = StandardScaler()
    _index = df.index
    _columns = df.columns
    
    X = ssd.fit_transform(df)
    df = pd.DataFrame(X,index=_index,columns=_columns)
  
  df = df.reset_index()
  df["time"] = pd.to_datetime(df["time"].astype(str))
  return df

def load_predict_Rad(cele="MSPP",ecode="ecmf001",resid2_name="lgb",drop_element=True,n_dim=3,isCNN=1):
    ESTIMATE1="/home/ysorimachi/data/synfos/som/model/estimate1"
    # path = f"{ESTIMATE1}/rad_{resid2_name}_{ecode}_{cele}.csv"
    path = f"{ESTIMATE1}/rad_{resid2_name}_{ecode}_{cele}_{n_dim}_{isCNN}.csv"
    # print(path)
    # sys.exit()
    df = pd.read_csv(path)
    use_col = ['time', 'OBS', 'MIX', 'SYN', 'EC', 'CR0','PRED1', 'RESID1', 'PRED2', 'CLUSTER','istrain']
    if not "istrain" in df.columns:
        df = train_flg(df)
    # print(df.head(30))
    # print(df.columns)
    # sys.exit()
    if drop_element:
        df = df[use_col]
    if df["time"].dtypes == object:
        df["time"] = pd.to_datetime(df["time"])
    df["dd"] = df["time"].apply(lambda x: x.strftime("%Y%m%d"))
    df["mm"] = df["time"].apply(lambda x: x.month)
    df["hh"] = df["time"].apply(lambda x: x.hour)
    
    return df
def load_predict_Rad2(cele="MSPP",ecode="ecmf001",resid2_name="lgb",drop_element=True,n_dim=3,isCNN=1):
  ESTIMATE3="/home/ysorimachi/data/synfos/som/model/estimate3"
  # path = f"{ESTIMATE1}/rad_{resid2_name}_{ecode}_{cele}.csv"
  path = f"{ESTIMATE3}/rad_{resid2_name}_{ecode}_{cele}_{n_dim}_{isCNN}.csv"
  # print(path)
  # sys.exit()
  df = pd.read_csv(path)
  use_col = ['time', f'PRED_{resid2_name}']
  # print(df.head(30))
  # print(df.columns)
  if df["time"].dtypes == object:
      df["time"] = pd.to_datetime(df["time"])
  # df["dd"] = df["time"].apply(lambda x: x.strftime("%Y%m%d"))
  # df["mm"] = df["time"].apply(lambda x: x.month)
  # df["hh"] = df["time"].apply(lambda x: x.hour)
  df = df[use_col]
  return df


if __name__ == "__main__":
  # mk_obs("201907","47891")
  # mk_fct("201907","47891")
  if 0:
    subprocess.run("rm -f {}".format(LOGFILE),shell=True)
    mk_dataset() #2021.05.05 calc...
  if 0:
    point_on_map()
    
  if 0:
    fd=1 #翌日予測
    _ecode,_scode,_name= load_10()
    #データの再作成
    for ecode in _ecode:
      mk_rad_210625(ecode=ecode,fd=fd)
      # sys.exit()
  
  if 1:
    load_weather_fcs()
    