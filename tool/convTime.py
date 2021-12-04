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
from datetime import datetime, timedelta


def dtinc(ctime12,N,delta, format_date=False):
  ini_time = pd.to_datetime(ctime12)
  int_delta=int(np.abs(delta))
  # if N==1: #year
    # delta_time = timedelta(year=np.abs(delta))
  # print(ini_time,N,int_delta)
  # sys.exit()
  if N==2: #month
    delta_time = timedelta(weeks=int_delta)
  elif N==3: #day
    delta_time = timedelta(days=int_delta)
  elif N==4: #hou
    delta_time = timedelta(hours=int_delta)
  elif N==5: #min
    delta_time = timedelta(minutes=int_delta)
  else:
    return "ERROR"

  if delta >=0:
    next_time = ini_time + delta_time
  else:
    next_time = ini_time - delta_time
  
  if format_date:
    return next_time
  else:
    return next_time.strftime("%Y%m%d%H%M")


if __name__ == "__main__":
  init="202007120930"
  time_test = dtinc(init,2,1)
  print(time_test)
