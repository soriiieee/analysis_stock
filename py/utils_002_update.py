#common
"""
log
init updates  2021.09.07 update - info mations
参考：
https://irbank.net/download
からcsvデータを取得して、決算情報等を更新して学習するときに便利なようにするprogram
"""
import os
# os.makedirs(new_dir_path_recursive, exist_ok=True)
import sys
from datetime import datetime,timedelta
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

import subprocess
import glob
import xlrd

HOME="/Users/soriiieee/work2/stock"
FUND_DAT = "/Users/soriiieee/work2/stock/dat/fundamental/origin" # 保存先
NAME_DAT = "/Users/soriiieee/work2/stock/dat/fundamental/names"  # 保存先
CONCAT_OUT="/Users/soriiieee/work2/stock/dat/fundamental/concat"



def update():
  """
  init : 2021.09.07
  main : fundamental 情報の取得
  """
  today = datetime.now().strftime("%Y%m%d")
  # ------------------------------
  # local functions ----
  def xls2csv():
    # path = f"{NAME_DAT}/data_j.xls"
    path = f"{HOME}/tmp/EdinetcodeDlInfo.csv"

    subprocess.run("nkf -w --overwrite {}".format(path), shell=True)
    return
  def mk_namelist():
    path = f"{NAME_DAT}/data_j.csv"
    df= pd.read_csv(path)
    print(df.head())
    print(df.columns)
    use_col = ['コード','銘柄名', '市場・商品区分', '33業種コード', '33業種区分', '17業種コード', '17業種区分',
       '規模コード', '規模区分']
    rename_col = ['code','name', 'market', 'code33', 'name33', 'code17', 'name17',
       'size', 'num_size']
    use_col2 = ['code','name', 'market','name33','name17','num_size']
    df = df[use_col]
    df.columns = rename_col
    df = df[use_col2]

  # 市場・商品区分 ['市場第一部（内国株）' 'ETF・ETN' 'JASDAQ(スタンダード・内国株）' 'JASDAQ(グロース・内国株）'
  # 'マザーズ（内国株）' 'PRO Market' '市場第二部（内国株）' '市場第一部（外国株）'
  # 'REIT・ベンチャーファンド・カントリーファンド・インフラファンド' 'JASDAQ(スタンダード・外国株）' 'マザーズ（外国株）'
  # '出資証券' '市場第二部（外国株）'] 13
  # 33業種コード ['50' '-' '6050' '2050' '3500' '1050' '9050' '3600' '3550' '5250' '3050'
  # '3250' '8050' '5050' '7200' '6100' '3800' '3100' '3650' '3400' '7100'
  # '3700' '3300' '3200' '3150' '3750' '3350' '3450' '7050' '7150' '5200'
  # '5100' '5150' '4050'] 34
  # 33業種区分 ['水産・農林業' '-' '卸売業' '建設業' '非鉄金属' '鉱業' 'サービス業' '機械' '金属製品' '情報・通信業' '食料品'
  # '医薬品' '不動産業' '陸運業' 'その他金融業' '小売業' 'その他製品' '繊維製品' '電気機器' 'ガラス・土石製品'
  # '証券、商品先物取引業' '輸送用機器' '石油・石炭製品' '化学' 'パルプ・紙' '精密機器' 'ゴム製品' '鉄鋼' '銀行業'
  # '保険業' '倉庫・運輸関連業' '海運業' '空運業' '電気・ガス業'] 34
  # 17業種コード ['1' '-' '13' '3' '7' '2' '10' '8' '5' '17' '12' '16' '14' '4' '9' '6'
  # '15' '11'] 18
  # 17業種区分 ['食品 ' '-' '商社・卸売 ' '建設・資材 ' '鉄鋼・非鉄 ' 'エネルギー資源 ' '情報通信・サービスその他 ' '機械 '
  # '医薬品 ' '不動産 ' '運輸・物流 ' '金融（除く銀行） ' '小売 ' '素材・化学 ' '電機・精密 ' '自動車・輸送機 '
  # '銀行 ' '電力・ガス '] 18
  # 規模コード ['7' '-' '4' '6' '2' '1'] 6
  # 規模区分 ['TOPIX Small 2' '-' 'TOPIX Mid400' 'TOPIX Small 1' 'TOPIX Large70'
  # 'TOPIX Core30'] 6
    # print(df.head())
    path2 = f"{NAME_DAT}/data_j2.csv"
    df.to_csv(path2, index=False)
    return
  # ------------------------------
  # main
  if 0: # fundamental csv ...
    subprocess.run("sh get_funda.sh {}".format(FUND_DAT), shell=True)
  if 0: # name(ticker)  & ETF list
    ##あまりうまくいかなかった
    # subprocess.run("sh get_tickers.sh {}".format(NAME_DAT), shell=True)
    """
     # 1: ここにアクセス(EDINET)
     https://disclosure.edinet-fsa.go.jp/E01EW/BLMainController.jsp?uji.verb=W1E62071InitDisplay&uji.bean=ee.bean.W1E62071.EEW1E62071Bean&TID=W1E62071&PID=currentPage&SESSIONKEY=1630917620447&kbn=2&ken=0&res=0&idx=0&start=null&end=null&spf1=1&spf2=1&spf5=1&psr=1&pid=0&row=100&str=&flg=&lgKbn=2&pkbn=0&skbn=1&dskb=&askb=&dflg=0&iflg=0&preId=1
     # 2: いかに保存(クリックする&zipの解凍)
     /Users/soriiieee/work2/stock/tmp
    """
    # xls2csv()
    mk_namelist()
  return 


def load_tbl():
  path2 = f"{NAME_DAT}/data_j2.csv"
  df = pd.read_csv(path2)
  return df

def tbl_columns(sel_col = None):
  # 業界一覧を確認するようのprogramになる
  # date: 2021.09.07
  df = load_tbl()
  if sel_col:
    use_col=[sel_col]
  else:
    use_col =['market', 'name33', 'name17', 'num_size']

  for c in use_col:
    print("-"*15,c,"-"*40)
    print(df[c].unique())
    print("-"*15)
    print(df[c].value_counts())
    # sys.exit()

def mk_gyoukai(col="name33",name="非鉄金属"):
  df = load_tbl()
  df = df[df[col]==name]
  return df


def mk_1y(_ticker):
  _list = sorted(glob.glob(f"{FUND_DAT}/fy*.csv"))
  for ticker in _ticker:
    # print(ticker)
    # sys.exit()
    for i,f in enumerate(_list):
      df = pd.read_csv(f,skiprows=1)
      df = df.reset_index()
      # print(df.columns)
      # sys.exit()
      df= df[df["コード"]==ticker]
      print(df.head())
      sys.exit()
      # df= df[df["コード"]==ticker]
      df= df[ticker]
      print(df.head())
    sys.exit()
    sys.exit()
    return

  return

def main():
  """
  main program
  date : 2021.09.07
  """
  if 0:
    tbl_columns(sel_col="name33") # 業界選定

  if 1:
    col ,name  = "name33", "非鉄金属"
    df = mk_gyoukai(col="name33",name="非鉄金属")
    _ticker = df["code"].values.tolist()
    
    mk_1y(_ticker)





if __name__ == "__main__":

  if 0:
    update()
  
  if 1:
    main()