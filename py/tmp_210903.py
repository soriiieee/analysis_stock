#common
"""
log
update 2021.09.03 init

"""
import os
# os.makedirs(new_dir_path_recursive, exist_ok=True)
import sys
from datetime import datetime 
import pandas as pd
import  numpy as np
import glob

#
def setting_simple_tbl():
  path = "../tbl/japan.csv"
  df = pd.read_csv(path)
  df = df.sort_values("code")
  use_col = ['code', 'name_x', 'market', 'category', 'is225', 'Ncate', 'cate_name2',
       'status', '単位']
  df = df[use_col]
  df.to_csv("../tbl/japan_simple.csv", index=False)

def get_etf():
  path = "../tbl/japan_simple.csv"
  df = pd.read_csv(path)
  # df = df[df["is225"]]
  return df

def ticker2name(ticker):
  tbl = get_etf()
  tbl = tbl[tbl["code"]==int(ticker)]
  if tbl.shape[0] !=0:
    name = tbl["name_x"].values[0]
  else:
    name = "NaN"
  return name

def get_csv(ticker):
  DIR="/Users/soriiieee/work2/sci/d0615/tmp"
  path = f"{DIR}/JP_{ticker}.csv"
  df = pd.read_csv(path)
  return df

def all_category():
  df = get_etf()
  print(df["category"].unique())
  return



def plot_ts():
  DIR="/Users/soriiieee/work2/sci/d0615/tmp"
  _path = sorted(glob.glob(f"{DIR}/JP_*.csv"))

  for p in _path:
    ticker = os.path.basename(p).split("_")[1][:4]
    name = ticker2name(ticker)

    ts = get_csv(ticker)
    print(ts.head(50))
    sys.exit()

    #plot function
    f,ax = plt.subplots(3,1,figsize=(15,8))
    for i, c in enumerate(["close","volume","log_diff"]):
      ax[i].plot(ts[c], label = c)
      ax[i].set_xlabel("time")
    f.savefig(f"../out/ts1/{name}.png",bbox_inches="tight")
    sys.exit()
  print(_path)

if __name__ == "__main__":
  if 1:
    all_category()
    sys.exit()

  if 1:
    # setting_simple_tbl()
    plot_ts()