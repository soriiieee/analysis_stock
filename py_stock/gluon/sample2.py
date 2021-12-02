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
# https://aws.amazon.com/jp/blogs/news/creating-neural-time-series-models-with-gluon-time-series/
https://ts.gluon.ai/tutorials/forecasting/quick_start_tutorial.html
"""
import mxnet as mx
from mxnet import gluon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.dataset.common import ListDataset
from gluonts.model.predictor import Predictor


from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.mx import Trainer
# ['constant', 'exchange_rate', 'solar-energy', 'electricity', 'traffic', 'exchange_rate_nips', 'electricity_nips', 'traffic_nips', 'solar_nips', 'wiki-rolling_nips', 'taxi_30min', 'kaggle_web_traffic_with_missing', 'kaggle_web_traffic_without_missing', 'kaggle_web_traffic_weekly', 'm1_yearly', 'm1_quarterly', 'm1_monthly', 'nn5_daily_with_missing', 'nn5_daily_without_missing', 'nn5_weekly', 'tourism_monthly', 'tourism_quarterly', 'tourism_yearly', 'm3_monthly', 'm3_quarterly', 'm3_yearly', 'm3_other', 'm4_hourly', 'm4_daily', 'm4_weekly', 'm4_monthly', 'm4_quarterly', 'm4_yearly', 'm5']
from gluonts.dataset.util import to_pandas
from gluonts.evaluation import make_evaluation_predictions #データ推論用のmodule群 
from utils_model import *
from pathlib import Path
# print(Path("./tmp"))
# sys.exit()
# print(list(dataset_recipes.keys()))

def cuntom_data():
  """[summary]

  Returns:
      [type]: Gluon_ts(DataSet format)　に直して学習&推論ができる形にして実施する予定
  """
  N=10
  T=100
  prediction_length = 24
  freq="1H"
  custom_dataset = np.random.normal(size=(N,T)) #データセットの数N=10/データセットの長さ=100
  # dataset (10,100) = (N,T)
  start = pd.Timestamp("01-01-2019", freq=freq) #2019-01-01 00:00:0
  # print(start)
  train = ListDataset([{ 'target': x,'start': start} for x in custom_dataset[:,:-prediction_length]],freq=freq) #100-24=76 data
  test = ListDataset([{ 'target': x,'start': start} for x in custom_dataset],freq=freq)#100 data
  return train,test

def get_est(dataset):
  estimator = SimpleFeedForwardEstimator(
      num_hidden_dimensions=[10],
      prediction_length=dataset.metadata.prediction_length,
      context_length=100,
      freq=dataset.metadata.freq,
      trainer=Trainer(
          ctx="cpu",
          epochs=5,
          learning_rate=1e-3,
          num_batches_per_epoch=100
      )
  )
  return estimator
  # print(custom_dataset[0,:10])
  # sys.exit()
def main():
  dataset = get_dataset("solar-energy", regenerate=False)
  train,test = get_data() #格納する箱を用意する
  estimator = get_est(dataset)
  
  train=False
  if train:
    predictor = estimator.train(dataset.train)
    predictor.serialize(Path("./model/"))
    # save_model("./predictor.pkl",predictor)
  else:
    predictor = Predictor.deserialize(Path("./model/")) #学習済モデルの読み込み
    
  forecast_it, ts_it = make_evaluation_predictions(
    dataset = dataset.test,
    predictor = predictor,
    num_samples = 100
  )
  
  forecasts,tss = list(forecast_it), list(ts_it)
  check(forecasts,tss ,n=3)
  # print(len(forecasts), len(tss))
  sys.exit()
  
  pass

def check(forecasts, tss,n=0):
  ts_entry = tss[n]
  ts_entry = np.array(ts_entry)
  print("ts_entry=>", ts_entry)
  return

if __name__ == "__main__":
  # check()
  main()

