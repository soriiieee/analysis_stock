# -*- coding: utf-8 -*-
#--------------------------------------------------------
#   DETAILES
#   program     :   getErrorValues
#   edit        :   yuichi sorimachi(20/01/06)
#   action      :   errors

#--------------------------------------------------------
#   module
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

#--------------------------------------------------------

# return me,rmse,nn
def me(x,y):
    df=pd.concat([x,y],axis=1)
    df = df.dropna()
    #calc
    if df.empty:
        err = 9999.
    else:
        err = np.sum(df.iloc[:,1] - df.iloc[:,0]) / df.shape[0]
        err = np.round(err,3)
    return err

def rmse(x,y):
    df= pd.concat([x,y],axis=1)
    df = df.replace(0,np.nan)
    df = df.dropna()
    #calc
    if df.empty:
        return 9999.
    else:
        err =np.sqrt(mean_squared_error(df.iloc[:,0],df.iloc[:,1]))
        return np.round(err,3)

def mae(x,y):
    df=pd.concat([x,y],axis=1)
    df = df.dropna()
    #calc
    if df.empty:
        err = 9999.
    else:
        err = np.sum(np.abs(df.iloc[:,1] - df.iloc[:,0])) / df.shape[0]
        err = np.round(err,3)
    return err

def r2(x,y):
    df=pd.concat([x,y],axis=1)
    df = df.dropna()
    #calc
    if df.empty:
        r2 = 9999.
    else:
        r2 = r2_score(df.iloc[:,1], df.iloc[:,0])
    return r2


def rmspe(x,y):
    """
    2021.07.26
    x: after(pd.Series)
    y: before(pd.Series)
    
    """
    df= pd.concat([x,y],axis=1)
    df = df.replace(0,np.nan)
    df = df.dropna()
    #calc
    if df.empty:
        return 9999.
    else:
        df["1"] = 1.
        df["ratio"] = df.iloc[:,1] / df.iloc[:,0]
        err =np.sqrt(mean_squared_error(df["ratio"],df["1"]))
        return np.round(err,3)

def mape(x,y):
    """
    2021.07.26
    x: after(pd.Series)
    y: before(pd.Series)
    """
    df=pd.concat([x,y],axis=1)
    df = df.dropna()
    #calc
    if df.empty:
        err = 9999.
    else:
        df["1"] = 1.
        df["ratio"] = df.iloc[:,1] / df.iloc[:,0]
        err = np.sum(np.abs(df["ratio"] - df["1"])) / df.shape[0]
        err = np.round(err,3)
    return err


def nrmse(x,y):
    df= pd.concat([x,y],axis=1)
    df = df.replace(0,np.nan)
    df = df.dropna()
    #calc
    if df.empty:
        return 9999.
    else:
        err = np.sqrt(mean_squared_error(df.iloc[:,0],df.iloc[:,1]))
        ave = np.mean(df.iloc[:,0])
        err /=ave
        return np.round(err,3)