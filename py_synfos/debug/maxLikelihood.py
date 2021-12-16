# -*- coding: utf-8 -*-
# when   : 2020.03.23 
# who : [sori-machi]
# what : [ ]
"""
・参考サイト　-> https://qiita.com/sato56513/items/212b4253fe8e15db1093
・scipy　gamma 関数について -> https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html


"""
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
# from getErrorValues import me,rmse,mae,r2 #(x,y)
from convSokuhouData import conv_sfc #(df, ave=minutes,hour)
# #amedas relate 2020.02,04 making...
from tool_AMeDaS import code2name, name2code, name2scode
# from tool_110570 import get_110570,open_110570
# from tool_100571 import get_100571,open_100571
#(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
import numpy as np
import scipy.stats as st
from statsmodels.base.model import GenericLikelihoodModel
# import holoviews as hv
# hv.extension('bokeh')
# from bokeh.plotting import show



def isFloat(x):
  try:
    return float(x)
  except:
    return np.nan

def clensing(df):
  df = df.replace(9999,np.nan)
  for c in df.columns:
    df[c].apply(lambda x: isFloat(x))
  return df

def load_data(month="202102",name="東京"):
  #-------load sfc Dataframe ---
  scode = name2scode(name)
  sfc_path = f"/work/ysorimachi/make_SOKUHOU3/out/{month}/sfc2/sfc_10minh_{month}_{scode}.csv"
  if os.path.exists(sfc_path):
    df = pd.read_csv(sfc_path)
    df = conv_sfc(df)
  else:
    df = pd.DataFrame()
  #-------個別案件---
  use_col = ["time",'windDirection', 'windSpeed', 'temp']
  if df.shape[0] !=0:
    df = df[use_col]
    df = df.set_index("time")
    df = clensing(df) #data clensing ---
  
  for c in df.columns:
    df[c] = df[c].apply(lambda x: np.nan if np.abs(x) > 100 else x)
  #-------個別案件---
  # names=["time","rrad","isyn","iecm","iecc","flg_ecm","flg_ecc"]
  return df


#----------------------------------------------------------
#----------------------------------------------------------
#----------------------------------------------------------
#----------------------------------------------------------
def F(X,a,b):
  """
  最尤推定を行いたい関数について
  """
  scale = 1/beta
  _p = st.gamma.pdf(X, a,loc=0, scale=scale)
  return _p


class GammaFit(GenericLikelihoodModel):
  
  def __init__(self,endog,exog=None,**kwds):
    if exog is None:
      exog = np.zeros_like(endog)
    super(GammaFit, self).__init__(endog, exog,**kwds)
  
  def nloglikeobs(self,params):
    a = params[0]
    b = params[1]
    
    #Loss 関数の表示 --->
    L = - np.log(F(self.endog,a=a,b=b))
    return L
  
  def fit(self,start_params= None, maxiter=10000,maxfun=5000,**kwds):
    m = super(GammaFit,self).fit(start_params = start_params, maxiter=maxiter, maxfun=maxfun)
    return m
  

def train():
  size_list = [100,500,1000]
  coeff_dic = {}
  
  for size in size_list:
    _a,_b = [],[]
    
    for i in range(100):
      
      data = st.gamma.rvs(2,loc = 10,scale=1, size=size)
      model = GammaFit(data) #推定したい確率分布を設定する
      a,b = 2,1
      
      res  = model.fit(start_params=[a,b])
      a,b = res.params[0] , res.params[1]
      
      _a.append(a)
      _b.append(b)
     




if __name__ == "__main__":
  main()
  # test2()
  