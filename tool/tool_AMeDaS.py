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
from getErrorValues import me,rmse,mae,r2 #(x,y)
from convSokuhouData import conv_sfc #(df, ave=minutes,hour)
#---------------------------------------------------
import subprocess
from conv2roma import conv2roma


def isint(s):
  try:
    int(s)
    return 1
  except ValueError:
    return 0

def code2name(code,res_type="name"):
  # path = "/home/griduser/work/make_SOKUHOU3/tbl/amdmaster_new.csv"
  path = "/home/ysorimachi/tool/amdmaster.index"
  use_col = ["Station Number","Station Name","Station Name.1"]
  renames = ["code","name","kana"]
  if not os.path.exists(path):
    pre_downloads()
  df = pd.read_csv(path)
  df =df.iloc[1:,:]
  df = df[use_col]
  df.columns = renames
  df["name"] = df["name"].apply(lambda x: x.strip())
  df["kana"] = df["kana"].apply(lambda x: x.strip())
  df["code"] = df["code"].apply(lambda x: str(int(x)) if isint(x)==1 else np.nan )
  df = df.dropna()
  df = df.drop_duplicates(subset=["name"],keep="last")
  df["roma"] = df["kana"].apply(lambda x : conv2roma(x))
  
  code = str(code)
  name = df.loc[df["code"]==code, res_type]
  
  if name.shape[0] !=0:
    return name.values[0]
  else:
    return "Nan"

def code2roma(code,res_type="roma"):
  # path = "/home/griduser/work/make_SOKUHOU3/tbl/amdmaster_new.csv"
  path = "/home/ysorimachi/tool/amdmaster.index"
  use_col = ["Station Number","Station Name","Station Name.1"]
  renames = ["code","name","kana"]
  if not os.path.exists(path):
    pre_downloads()
  df = pd.read_csv(path)
  df =df.iloc[1:,:]
  df = df[use_col]
  df.columns = renames
  df["name"] = df["name"].apply(lambda x: x.strip())
  df["kana"] = df["kana"].apply(lambda x: x.strip())
  df["code"] = df["code"].apply(lambda x: str(int(x)) if isint(x)==1 else np.nan )
  df = df.dropna()
  df = df.drop_duplicates(subset=["name"],keep="last")
  df["roma"] = df["kana"].apply(lambda x : conv2roma(x))

  name = df.loc[df["code"]==code, res_type]
  if name.shape[0]==1:
    return name.values[0]
  else:
    return "Nan"

def name2code(name="name"):
  # path = "/home/griduser/work/make_SOKUHOU3/tbl/amdmaster_new.csv"
  path = "/home/ysorimachi/tool/amdmaster.index"
  use_col = ["Station Number","Station Name","Station Name.1"]
  renames = ["code","name","kana"]
  if not os.path.exists(path):
    pre_downloads()
  df = pd.read_csv(path)
  df =df.iloc[1:,:]
  df = df[use_col]
  df.columns = renames

  df["name"] = df["name"].apply(lambda x: x.strip())
  df["kana"] = df["kana"].apply(lambda x: x.strip())

  df["code"] = df["code"].apply(lambda x: str(int(x)) if isint(x)==1 else np.nan )
  df = df.dropna()
  df = df.drop_duplicates(subset=["name"],keep="last")
  df["roma"] = df["kana"].apply(lambda x : conv2roma(x))

  
  code = df.loc[df["name"]==name, "code"]
  if code.shape[0]==0:
    return "nan"
  else:
    return code.values[0]

def amedas_table(outpath=None):
  path = "/home/ysorimachi/tool/amdmaster.index"
  if not os.path.exists(path):
    pre_downloads()
  df = pd.read_csv(path)
  df = df.iloc[1:,:]

  df["Station Number"] = df["Station Number"].apply(lambda x: np.nan if isint(x)==0 else int(x))
  df = df.dropna(subset=["Station Number"])
  df = df.drop_duplicates(subset=["Station Number"],keep="last")
  for c in ['Latitude','Latitude.1','Longitude','Longitude.1']:
    df[c] = df[c].astype(float)
  
  df["Latitude"] = np.round(df['Latitude']  + df['Latitude.1'] /60.,4)
  df["Longitude"] = np.round(df['Longitude']  + df['Longitude.1'] /60.,4)
  
  use_col =['Station Number', 'Station Name', 'Station Name.1', 'Station Name.2',
       'Latitude','Longitude']
  
  df = df[use_col]
  df.columns = [ 'code', 'kanji', 'kana', 'name','lat','lon']
  if path:
    df.to_csv(outpath, index=False)
    # print(outpath)
  
  # sys.exit()
    
  com =f'sh tmp.sh'
  subprocess.run(com,cwd="./",shell=True)
  return

def pre_downloads():
  com="wget http://www.data.jma.go.jp/obd/stats/data/mdrr/chiten/meta/amdmaster.index"
  subprocess.run(com,cwd="./", shell=True)
  subprocess.run("nkf -w --overwrite amdmaster.index" ,cwd="./", shell=True)
  return


def acode2scode(code="11016", with_name=False):
  """
  2021.07.30 sorimachi making ...
  """
  tbl = pd.read_csv("/home/ysorimachi/work/make_SOKUHOU3/tbl/list_sokuhou.csv")
  tbl=tbl[tbl["code"] == int(code)]
  
  if tbl.empty:
    return 99999
  else:
    scode = tbl["scode"].values[0]
    name = tbl["name"].values[0]
    if with_name:
      return [scode,name]
    else:
      return scode

def name2scode(name):
  acode = name2code(name)
  if acode != "nan":
    scode = acode2scode(acode)
    if scode != "nan":
      return scode
    else:
      return f"nan -> acode[{acode}]"
  else:
    return f"nan -> acode[{acode}]"
  
  
def scode2name(scode, res_type="roma"):
  if type(scode) == int:
    scode = str(scode)
  
  acode = scode2acode(scode)
  if acode != 99999:
    name = code2name(acode,res_type = res_type)
    
    if name != "Nan":
      return name
    else:
      return f"nan -> name[{name}]"
  else:
    return f"nan -> name[{name}]"


def scode2acode(code="47401", with_name=False):
  """
  2021.07.30 sorimachi making ...
  """
  tbl = pd.read_csv("/home/ysorimachi/work/make_SOKUHOU3/tbl/list_sokuhou.csv")
  tbl=tbl[tbl["scode"] == int(code)]
  
  if tbl.empty:
    return 99999
  else:
    acode = tbl["code"].values[0]
    name = tbl["name"].values[0]
    if with_name:
      return [acode,name]
    else:
      return acode


if __name__== "__main__":
  # code= acode2scode(code=11016, with_name=True)
  code= scode2acode(code=47401, with_name=False)
  print(code)

