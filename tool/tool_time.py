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


def dtinc(ini_time,part,n):
  ini_time = pd.to_datetime(ini_time)
  if part==3:
    delta = timedelta(days=n)
  elif part==4:
    delta = timedelta(hours=n)
  elif part==5:
    delta = timedelta(minutes=n)

  conv_time = ini_time + delta
  return conv_time.strftime("%Y%m%d%H%M")


if __name__=="__main__":
  t = dtinc("201812010000",5,-50)
  print(t)