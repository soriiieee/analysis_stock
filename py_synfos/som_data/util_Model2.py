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

#---------------------------------------------------
import subprocess
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# from torch.utils.data import DataLoader, TensorDataset, Dataset
# from torchvision import transforms
# from torchinfo import summary #torchinfoはニューラルネットの中身を見れるのでおすすめ


seed=1
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
#fit tool
from sklearn.linear_model import  ElasticNet,Lasso,Ridge
from sklearn.svm import SVR #{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

import lightgbm as lgb #light gbm
#test data 
from sklearn.datasets import load_wine,load_boston
# https://scikit-learn.org/stable/modules/classes.html?highlight=datasets#module-sklearn.datasets

import pickle

class Resid2:
    def __init__(self,name, pretrained=False):
        self.name = name
        if not pretrained:
            if name == "lasso":
                self.m = Lasso()
            elif name == "ridge":
                self.m = Ridge()
            elif name == "svr":
                # method = ""
                # self.m = SVR(method=method)
                self.m = SVR()
            elif name == "tree":
                self.m = DecisionTreeRegressor()
            elif self.name =="rf":
                self.m = RandomForestRegressor()
            elif self.name =="lgb":
                # self.m = 
                self.m = None
            elif self.name == "mlp3":
                self.m = MLPRegressor(hidden_layer_sizes=(200,100,50,),random_state=1, shuffle=True,verbose=False)
            elif self.name == "mlp5":
                self.m = MLPRegressor(hidden_layer_sizes=(300,200,100,50,30),random_state=1, shuffle=True,verbose=False)
            else:
                sys.exit(["lasso","ridge","svr","tree","rf","lgb","mlp3","mlp5"])
        else:
            self.m = self.load(pretrained)
            
    def fit(self,X,y):
        if self.name != "lgb":
            self.m.fit(X,y)
        else:
            x0,x1,y0,y1 = train_test_split(X,y,test_size = 0.2, random_state = 123)
            train_sets = lgb.Dataset(x0,y0)
            eval_sets = lgb.Dataset(x1,y1)
            #https://lightgbm.readthedocs.io/en/latest/Parameters.html -> params の内容が記載
            params = {
                # 'objective': 'reg:squarederror',
                "learning_rate": 0.1,       # 学習率(default 0.1)
                "metric": "root_mean_squared_error", # モデルの評価指標
                "seed": 42
            }
            self.m = lgb.train(params,train_sets,valid_sets = [train_sets,eval_sets],
                        num_boost_round=1000,early_stopping_rounds=20)
        return 
    
    def predict(self,X):
        return self.m.predict(X)
    
    def save(self,path):
        with open(path,"wb") as pkl:
            pickle.dump(self.m,pkl)
        return
    
    def load(self,path):
        with open(path,"rb") as pkl:
            model = pickle.load(pkl)
        return model
    
    
    #特殊メソッドの上書
    def __repr__(self):
        return f"model -> {self.m}"
    
    
def main():
    dataset = load_boston()
    y = pd.Series(dataset.target,name="target")
    X = pd.DataFrame(dataset.data,columns=dataset.feature_names)
    df = pd.concat([X,y],axis=1)
    
    
    # m = Model2("lgb",pretrained=False)
    # m.fit(X,y)
    path = "./lgb.pkl"
    m = Resid2("lgb",pretrained=path)
    df["pred"] = m.predict(X)
    
    print(df[["target","pred"]])
    sys.exit()
    
    
if __name__=="__main__":
    main()
    