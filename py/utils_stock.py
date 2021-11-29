#common
"""
log
init 2021.06.15 17:00 stock data templates
update 2021.06.26 11:00 stock data - データ取得および、解析
update 2021.09.03 update
"""
import os
# os.makedirs(new_dir_path_recursive, exist_ok=True)
import sys
from datetime import datetime 
import pandas as pd
import  numpy as np
import subprocess
from tqdm import tqdm
#science 
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.integrate._ivp.radau import P
# from scipy.integrate._ivp.radau import P
from sklearn.preprocessing import MinMaxScaler


#jisaku
# from config import Com
# HOME=Com.HOME  2021.06.17
HOME="/Users/soriiieee/work2/stock" # 2021.09.03

print("START...Now is ->",datetime.now())
# print(Com.HOME)
import yfinance as yf
#for japanese stocks
from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError



"""
参考にしたサイト(数値積分)
stock list
https://towardsdatascience.com/how-to-get-stock-data-using-python-c0de1df17e75
"""
# - subroutine
def clensing_yf(df,NATION):
  if NATION=="JP":
    # print("debug 21.09.03")
    df["diff_max"] = df["high"] - df["low"]
    df["diff_day"] = df["close"] - df["open"]

    # print(df.head())
    # sys.exit()
    use_col = ["time","open","close","high","low","diff_max","diff_day","volume"] #2021.09.11
    df = df[use_col]
    # df.columns =["time","close","diff_max","diff_day","volume"]
    df["log_diff"] = np.log(df["close"]).diff()
    df = df.round(4)
    return df

  if NATION=="US":
    df = df.reset_index()
    df["diff_max"] = df["High"] - df["Low"]
    df["diff_day"] = df["Close"] - df["Open"]

    use_col = ["Date","Close","diff_max","diff_day","Volume"]
    df = df[use_col]
    df.columns = ["time","close","diff_max","diff_day","volume"]
    df["log_diff"] = np.log(df["close"]).diff()
    df = df.round(4)
    return df

def get(ticker = 'MSFT',NATION="US"):
  # path = f"../tmp/{NATION}_{ticker}.csv"
  # if not os.path.exists(path):
  if 1:
    """tmpの中にファイルがない場合は取得する"""
    # print("Not Founded Now get! ..")
    if NATION=="US":
      #get data on this ticker
      data = yf.Ticker(ticker)
      #get the historical prices for this ticker
      # tickerDf = data.history(period='1d', start='2010-1-1', end='2020-1-25')
      df = data.history(period='1d', start='2015-1-1')
      # print(df.tail())
      df = clensing_yf(df,"US")
      return df
    
    if NATION=="JP":
      # ticker = "7203"
      """
      日本株のhistorical　dataを取得する物語
      https://myfrankblog.com/stock_price_with_yahoo_finance_api2/#i
      """
      if not ".T" in ticker:
        ticker = f"{ticker}.T"
      
      # print(datetime.now(), "[DEBUG]")
      # print(ticker)
      # sys.exit()
      myshare = share.Share(ticker)

      try:
        dat = myshare.get_historical(
          share.PERIOD_TYPE_YEAR,100,
          share.FREQUENCY_TYPE_DAY,1)
      except YahooFinanceError as e:
        print(e.message)
        dat=None
      
      if dat:
        df = pd.DataFrame(dat)
        df["time"] = pd.to_datetime(df["timestamp"],unit="ms")
        # print(df.head())
      df = clensing_yf(df,"JP")
    # df.to_csv(path, index=False)
      return df

  else:
    if check_dt(path) == 0:
      print("already getted ..check dt==0")
      df = pd.read_csv(path)
      return df
    else:
      print("already getted ..check dt==1")
      os.remove(path)
      get(ticker = ticker,NATION=NATION)
      return df

def plot_df(df,_col):
  """簡易的にデータ確認を行う"""
  f,ax = plt.subplots(figsize= (22,8))
  for c in _col:
    ax.plot(df[c], label=c)
  ax.legend()
  f.savefig("../out/tmp/data.png",bbox_inches="tight")
  plt.close()
  return 

def pre_porocess(ticker = 'MSFT',NATION="US", isNORM=True):
  df = get(ticker,NATION)
  if isNORM:
    mms = MinMaxScaler()
    df = df.dropna()
    df = df.set_index("time")
    _time = df.index
    mms_use_col = ["close","volume","log_diff"]
    X = mms.fit_transform(df[mms_use_col])
    print(df.shape)
    df = pd.DataFrame(X,columns = mms_use_col)
    df.index = _time
  return df

def compare():
  # df = pre_porocess(ticker = 'MSFT',NATION="US", isNORM=True)
  list_JP(cate='輸送用機器', N=10)

  # df = pre_porocess(ticker = '7203',NATION="JP", isNORM=True)
  df = pre_porocess(ticker = '^N255',NATION="JP", isNORM=True)

  print(df.head())
  return 

def list_JP(cate=None, N=5,isLIST=False):
  """
  cate: ['水産・農林業' '鉱業' '建設業' '食料品' 'サービス業' '卸売業' '小売業' '繊維製品' '電気機器' '不動産業' '化学'
 '金属製品' 'パルプ・紙' '医薬品' '精密機器' '情報・通信' '石油・石炭製品' 'ゴム製品' 'ガラス・土石製品' '鉄鋼' '機械'
 '非鉄金属' '輸送用機器' '銀行業' 'その他製品' 'その他金融業' '証券業' '保険業' '陸運業' '海運業' '空運業'
 '倉庫・運輸関連業' '電気・ガス業']
  """
  # cate = '輸送用機器'
  path = f"../tbl/japan.csv"
  df  = pd.read_csv(path)
  df = df.sort_values("code")
  use_col = ['code', 'name_x', 'market', 'category', 'is225','sales_2020']
  df = df[use_col]
  if cate:
    df = df[(df["is225"]==1)&(df["category"]==cate)]
  else:
    df = df[df["is225"]==1].reset_index(drop=True)
  df = df.sort_values('sales_2020', ascending=False)
  df = df.reset_index(drop=True)
  if N:
    df = df.iloc[:N,:]
  if isLIST:
    _tic = df["code"].values.tolist()
    _name = df["name_x"].values.tolist()
    return _tic,_name
  else:
    return df

def plot_10com():
  _com = []
  _cate = ['非鉄金属','輸送用機器','銀行業','情報・通信']
  for cate in _cate:
    com = list_JP(cate=cate, N=9,isLIST=False)
    print(com)
    sys.exit()
    _com.append(com["sales_2020"])

  f,ax = plt.subplots(figsize=(22,8))
  df = pd.concat(_com,axis=1)
  df.columns = _cate
  print(df)
  sys.exit()

  for cate in _cate:
    ax.plot(df[cate],label=cate)
  ax.legend()
  f.savefig(".sample225.png",bbox_inches="tight")
  # print(df.head())
  return


def main():
  _ticker,_name = list_JP(cate='輸送用機器', N=9,isLIST=True)
  # print(_ticker,_name)
  # sys.exit()
  """
  ETF一覧(2021.09.03):
  https://www.jpx.co.jp/equities/products/etfs/issues/01.html
  1321 ：日経平均225
  """
  _df =[]
  df = pre_porocess(ticker = str(1321),NATION="JP", isNORM=False) #nikkei 
  _df.append(df)
  for ticker,name in zip(_ticker,_name):
    # df = pre_porocess(ticker = str(ticker),NATION="JP", isNORM=True)
    # print(df.head())
    pass
    # sys.exit()
  return

def check_dt(path = "../tmp/JP_7203.csv"):
  unix_t = os.stat(path).st_mtime
  delta = datetime.now() - datetime.fromtimestamp(unix_t)
  # print(dir(delta))
  if delta.days>2:
    return 1
  else:
    return 0

def clean_csv():
  #保存先のデータを新しくする
  TMP="../tmp"
  subprocess.run(f"rm -f {TMP}/*.csv",shell=True)
  return

def load_japan():
  path = "../dat/fundamental/names/data_j2.csv"
  return pd.read_csv(path)

def code2name(ticker=1301):
  """
  2021.09.11 
  日本株専用　tickerをnameに変換する 
  """
  df = load_japan()
  _code = df["code"].values.tolist()
  df = df[["code","name"]]
  df_dict = df.set_index("code").to_dict()["name"]
  if ticker in _code:
    return df_dict[ticker]
  else:
    return "NO-NAME"

def isFloat(x):
  try:
    v = float(x)
  except:
    v = np.nan
  return v

def clensing(df,_col):
  for c in _col:
    df[c] = df[c].apply(lambda x: isFloat(x))
  return df



if __name__ == "__main__":
  if 0:
    compare()
  
  if 0:
    clean_csv()
  if 0:
    # plot_10com()
    main()
    # check_dt()
  
  if 1:
    name = code2name()
    # print(name)