# -*- coding: utf-8 -*-
# when   : 2021.04.12
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
#---------------------------------------------------
# sori -module
sys.path.append('/home/ysorimachi/tool')
from utils_plotly import plotly_1axis #(df,_col,html_path,title="sampe")
from tool_time import dtinc
#(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
import subprocess
import xml.etree.ElementTree as etree
from utils_log import log_write

DHOME="/work2/ysorimachi/synfos/data/SYNFOS-solar_ver2/recalc_hkrk/1"
_DIR=sorted(os.listdir(DHOME))

def count_file():
  for DIR in _DIR:
    CHECK_DIR=os.path.join(DHOME,DIR)
    subprocess.run(f"echo {DIR} `ls {CHECK_DIR} | wc`",shell=True)
  return


#------------
count_file()

  