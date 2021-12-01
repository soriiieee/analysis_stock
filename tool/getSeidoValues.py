# -*- coding: utf-8 -*-
# when   : 2020.0x.xx
# who : [sori-machi]
# what : [ ]
#---------------------------------------------------------------------------
# basic-module
# import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
import numpy as np
#---------------------------------------------------
# sori -module
sys.path.append('/home/griduser/tool')
# from getErrorValues import me,rmse,mae,r2 #(x,y)

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
# initial
#

class SeidoValues:
  def __init__(self,x,y,minmax=None):
    tmp = pd.concat([x,y],axis=1)
    _col = tmp.columns

    tmp = tmp.replace(0, np.nan)
    tmp = tmp.dropna()
    if minmax:
      tmp = tmp[tmp[_col[0]]> minmax[0]]
      tmp = tmp[tmp[_col[0]]< minmax[1]]

    self.tmp = tmp

    #cleaning

    # print("make instance")
  def corr(self,x,y):
    if self.tmp.shape[0] !=0:
      corr = np.corrcoef(self.tmp.values[:,0],self.tmp.values[:,1])[0, 1]
    else:
      corr = 9999.
    return corr
