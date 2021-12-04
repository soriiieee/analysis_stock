#common
"""
log
init updates  2021.09.07 update - info mations
init updates  2021.09.11 update - info mations
init updates  2021.09.17 update
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
import time

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

try:
  from utils_001_stock import code2name,clensing, get
except:
  sys.path.append("..")
  from py.utils_001_stock import code2name,clensing, get


# -　環境ごとに設定を変える　--#
if os.path.exists("/Users/soriiieee/work2/stock"):
  HOME="/Users/soriiieee/work2/stock"
else:
  HOME="/home/ysorimachi/work/sori_py2/analysis_stock"
  
FUND_DAT = f"{HOME}/dat/fundamental/origin" # 保存先
NAME_DAT = f"{HOME}/dat/fundamental/names"  # 保存先
CONCAT_OUT=f"{HOME}/dat/fundamental/concat"

def fundamental(yy="2020"):
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

    path2 = f"{NAME_DAT}/data_j2.csv"
    df.to_csv(path2, index=False)
    return
  # ------------------------------
  # main
  if 1: # fundamental csv ...
    subprocess.run("sh get_funda.sh {} {}".format(FUND_DAT,yy), shell=True)
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

def master_fundamental():
  """
  2021.11.09 sorimachi setting ---
  """
  _yy = sorted(os.listdir(FUND_DAT))
  df = load_tbl()
  # print(df.shape) #(4131, 6)
  # print(df.head())
  # print(check_cate("name17"))
  _ticker = df["code"].astype(int).values.tolist()
  for yy in _yy:
    mk_1y(_ticker,yy=yy,save=True)
    print(datetime.now(),"[end]", yy)
  return 

def load_tbl():
  """2021.11.14 update(空白文字の削除)"""
  path2 = f"{NAME_DAT}/data_j2.csv"
  df = pd.read_csv(path2)
  for c in ['name', 'market', 'name33', 'name17', 'num_size']:
    df[c] = df[c].apply(lambda x: x.replace(" ","").replace(" ",""))
  return df



def check_cate(cate = None):
  # 業界一覧を確認するようのprogramになる
  # date: 2021.09.07
  # date: 2021.11.14
  if cate is None:
    sys.exit("['market', 'name33', 'name17', 'num_size']")
  
  df = load_tbl()

  _uniq = df[cate].unique()
  _n=[]
  for u in _uniq:
    tmp = df[df[cate]==u]
    n_tmp = tmp.shape[0]
    _n.append(n_tmp)

  print("-"*15,cate,"-"*15)
  _list = [ f"{u}({n})" for u,n in zip(_uniq,_n)]
  print(_list)
  return 

def mk_1y(_ticker,yy="2021",save=True):
  _list = sorted(glob.glob(f"{FUND_DAT}/{yy}/fy*.csv"))

  #_make category list
  _df=[]

  all_col = ['総資産', '純資産', '株主資本', '利益剰余金', '短期借入金', '長期借入金', 'BPS', '自己資本比率','営業CF', '投資CF', '財務CF', '設備投資', '現金同等物', '営業CFマージン', '売上高','営業利益', '経常利益', '純利益', 'EPS', 'ROE', 'ROA','一株配当', '剰余金の配当','自社株買い', '配当性向', '総還元性向', '純資産配当率']
  sub_col = ['純資産', '株主資本', '利益剰余金', '短期借入金', '長期借入金', 'BPS', '自己資本比率','営業CF', '投資CF', '財務CF', '売上高','営業利益', '経常利益', '純利益', 'EPS', 'ROE', 'ROA','一株配当', '剰余金の配当']

  for i,f in enumerate(_list):
    df = pd.read_csv(f,skiprows=1)
    if i>0:
      df = df.drop("年度",axis=1)
    df= df.loc[df["コード"].isin(_ticker),:]
    df["コード"] = df["コード"].astype(int)
    df = df.set_index("コード")
    _df.append(df)
  
  df = pd.concat(_df,axis=1)
  _name = [ code2name(x) for x in df.index ]
  df["name"] = _name
  # df = clensing(df,_col = sub_col)
  df = clensing(df,_col = all_col)
  df.index.name = "code"
  df = df.reset_index()
  df = df.sort_values('code',ascending=True)

  if save:
    df.to_csv(f"{FUND_DAT}/master/funda_set_{yy}.csv", index=False)
  return df


def mk_list(col="name33",name ="輸送用機器"):
  """
  main program
  date : 2021.09.07
  """
  def mk_gyoukai(col="name33",name="非鉄金属"):
    
    df = load_tbl()
    df = df[df[col]==name]
    return df

    # 業界からそのticker listを引っ張ってくるような, sub-routine
  df = mk_gyoukai(col=col,name=name)
  _ticker = df["code"].values.tolist()
  #-------------------
  # 売上の上下でsortして表示するようなprogram
  df = mk_1y(_ticker,yy="2021")
  return df

def check_com(col,name, n=10, download=True):
  """
  2021.11.14 data downloads Japan data
  """

  df = mk_list(col=col,name =name)

  if df.shape[0] ==0:
    sys.exit("DataFrame Is None! please re-input col & name !")

  # データ格納用のdhirectorの生成を実施する
  OUT_DIR=f"/Users/soriiieee/work2/stock/out/ts1/{col}/{name}"
  os.makedirs(OUT_DIR,exist_ok=True)

  _code = df["code"].values.tolist()
  _name = df["name"].values.tolist()
  
  if n:
    _code=_code[:n]
    _name=_name[:n]
  
  if download:
    for i,(code,name) in enumerate(zip(_code,_name)):
      df = get(ticker = str(code),NATION="JP")
      df.to_csv(f"{OUT_DIR}/{code}_{name}.csv", index=False)
      print(datetime.now(), "[END]", i,name)
      time.sleep(0.2) #アクセス過多を防ぐための操作
  print("outfile is -->")
  print(OUT_DIR)


if __name__ == "__main__":

  if 1: #年間に１回ほど、更新する企業の規模感fundamental データのダウンロードの実施作業
    # for yy in ["2018","2017"]:
    #   fundamental(yy=yy)
      # sys.exit()
    master_fundamental()
  
  if 0: #業界別の企業一覧検索
    check_cate(cate="name17") #['market', 'name33', 'name17', 'num_size']
    # sys.exit()
    
    col ,name= "name17","商社・卸売 "
    check_com(col=col,name =name,n=20)
