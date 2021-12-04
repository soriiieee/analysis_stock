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
# import warnings
# warnings.simplefilter('ignore')
# from tqdm import tqdm
# import seaborn as sns
#---------------------------------------------------
# sori -module
sys.path.append('/home/griduser/tool')
# from getErrorValues import me,rmse,mae,r2 #(x,y)
# from getPlot import drawPlot,autoLabel #(x=False,y,path),(rects, ax)
# from convSokuhouData import conv2allwithTime,conv2CutCols #(df),(df, ave=minutes,hour)

def checkFile(input_path):
  flg = True
  # file ari/ nashi
  if  os.path.exists(input_path):
    pass
  else:
    return False
  
  # file size
  if os.path.getsize(input_path):
    pass
  else:
    return False

  return flg


if __name__ =="__main__":
  file_name="amd_10minh_201612_18174.csv"
  input_path = f"/work/griduser/tmp/ysorimachi/snowdepth_calc200525/dat0701/201612/{file_name}"
  print(checkFile(input_path))