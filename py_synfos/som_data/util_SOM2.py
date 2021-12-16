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


# sys.path.append('/home/ysorimachi/work/sori_py2/deepl/py')
sys.path.append('/home/ysorimachi/work/ecmwf/py_som')
from som2 import SOM
from sklearn.cluster import KMeans,AgglomerativeClustering
# from (self, m=3, n=3, dim=3, lr=1, sigma=1,distance="ED",max_iter=3000, epoch=1)
class ClusterModel:
    def __init__(self,method,n_clusters=16,DIMENTION=128,max_iter=20000,epoch=10):
        self.method = method
        map_size = int(np.sqrt(n_clusters))
        
        if method=="kmeans":
            m = KMeans(n_clusters=n_clusters)
        if method=="ward":
            m = AgglomerativeClustering(n_clusters=n_clusters)
        if method =="SOM-ED":
            m = SOM(m=map_size, n=map_size,dim=DIMENTION,distance="ED",max_iter=max_iter, epoch=epoch)
        if method == "SOM-SSIM":
            m = SOM(m=map_size, n=map_size,dim=DIMENTION,distance="SSIM",max_iter=max_iter,epoch=epoch)
        self.model = m
        self.isTrained = 0
        
    def fit(self,X):
        self.model.fit(X)
        self.isTrained = 1
        self.X = X
        
    @property
    def labels_(self):
        if self.method=="kmeans" or self.method=="ward":
            lbl = pd.Series(self.model.labels_,name = "labels")
            lbl = self.model.labels_
        else:
#             print("SOM")
            lbl = self.model.predict(self.X)
            # lbl = pd.Series(lbl,name="labels")
        return lbl
      
    @property
    def cluster_centers_(self):
        if self.method=="kmeans" or self.method=="ward":
            lbl = pd.Series(self.model.labels_,name = "labels")
        else:
#             print("SOM")
            lbl = self.model.predict(self.X)
            lbl = pd.Series(lbl,name="labels")
        return self.model.cluster_centers_ 
        
    def predict(self,X):
        if self.isTrained == 1:
            pred = self.model.predict(X)
        else:
            self.fit(X)
            pred = self.model.predict(X)
            
        pred = pd.Series(pred,name = "pred")
        return pred