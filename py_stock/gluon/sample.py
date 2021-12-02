# -*- coding: utf-8 -*-
# when   : 2021.11.10
# who : [sori-machi]
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
#amedas relate 2020.02,04 making...
# from tool_AMeDaS import code2name, name2code
# from tool_110570 import get_110570,open_110570
# from tool_100571 import get_100571,open_100571
#(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
import subprocess
outd ='/home/griduser/work/sori-py2/deep/out/0924_01'
# os.makedirs(outd, exist_ok=True)
# subprocess.run('rm -f *.png', cwd=outd, shell=True)

"""
時系列予測(単変数)にて実装を行う予定
https://aws.amazon.com/jp/blogs/news/creating-neural-time-series-models-with-gluon-time-series/
"""
from gluonts.model.deepar import DeepAREstimator
try:
  from gluonts.trainer import Trainer
except:
  from gluonts.mx.trainer import Trainer

from gluonts.dataset.common import ListDataset
from itertools import islice
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.dataset.util import to_pandas

from utils_plot import *
from utils_data import *
from utils_model import *


def plot_forecasts(tss,forecasts, past_length,num_plots, png_path="./plt_forecasts.png"):
  """[summary] 2021.11.10
  将来予測の値を出力してpngで書き出すprogramの表示

  Args:
      tss ([type]): [description]
      forecasts ([type]): [description]
      past_length ([type]): [description]
      num_plots ([type]): [description]
  """
  f,ax = plt.subplots(num_plots, 1,figsize=(22,18))
  for j,(target,forecast) in enumerate(islice(zip(tss,forecasts), num_plots)):
    ax[j].plot(target[-past_length:], linewidth=2)
    ax[j].grid(which="both")
    ax[j].legend(["observations", "median prediction", "90% confidence interval", "50% confidence interval"])
    # forecast.plot(color='g')
    fcs = to_pandas(forecast)
    print(fcs.head())
    sys.exit()
    # print(type(forecast))
    # fcs = pd.DataFrame(forecast.values)
    # print(fcs.head())
    # sys.exit()
  f.savefig(png_path, bbox_inches="tight")
  return

def check():
  """[ 2021.11.10 -> ]
  ・
  """
  df = load_amazon_twitter().iloc[:200,:]
  # print(df.head())
  # print(df.tail())
  plot_ts(df,c ="value")
  return 

def main():
  estimator = DeepAREstimator(freq = "5min", prediction_length=36, trainer= Trainer(epochs=10))
  # print(estimator)
  
  df = load_amazon_twitter()
  # print(df.index[0])
  # print(df.value[:"2015-04-05 00:00:00"])
  # sys.exit()
  train_data = ListDataset([{"start":df.index[0],"target":df.value[:"2015-04-05 00:00:00"]}], freq="5min")

  train=False
  if train:
    predictor = estimator.train(training_data = train_data)
    save_model("./predictor.pkl",predictor)
  else:
    predictor = load_model("./predictor.pkl")
  # dataの形式はシンプル!　index->に
  # 3 ----- 可視化ツールで見てみる!
  test_data = ListDataset([
    {"start":df.index[0],"target":df.value[:"2015-04-10 03:00:00"]},
    {"start":df.index[0],"target":df.value[:"2015-04-15 18:00:00"]},
    {"start":df.index[0],"target":df.value[:"2015-04-20 12:00:00"]}
  ],freq="5min")
  
  forecast_it, ts_it = make_evaluation_predictions(test_data, predictor=predictor, num_samples=100)
  forecasts = list(forecast_it)
  tss = list(ts_it)
  # print(test_data)
  # print(len(forecasts))
  # sys.exit()
  
  png_path = "./plt_forecasts.png"
  plot_forecasts(tss, forecasts, past_length=150, num_plots=3,png_path=png_path)
  return


if __name__ == "__main__":
  # check()
  main()

