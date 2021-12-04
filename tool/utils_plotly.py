# basic-module
import matplotlib.pyplot as plt
import sys, os,re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.simplefilter('ignore')

sys.path.append('/home/ysorimachi/tool')
from getErrorValues import me,rmse,mae,r2 #(x,y)
# from plotMonthData import plotDayfor1Month #(df,_col,title=False)
from reduceMemory import reduce_mem_usage
sys.path.append("/home/ysorimachi/work/hokuriku/py")

import plotly
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def plotly_1axis(df,_col,html_path,title="sampe"):
  fig = px.line(df,x='time', y=_col, title=title)
  
  # fig.update_yaxes(range=[0,1000], title="GSM[W/m2]") #2021.05.26 for eimu(GSM)
  fig.update_yaxes(range=[0,1400], title="日射量[W/m2]") #2021.06.30 for eimu(GSM)
  fig.write_html(html_path)
  return 


def plotly_2axis(df,col1,col2,html_path, title="sampe",vmax=1000):
  """
  #2021.07.21 for hokuriku teleme 
  update 2021.09.09
  """
  # _trace=[]
  fig = make_subplots(specs=[[{"secondary_y": True}]])
  
  for c in col1:
    fig.add_trace(go.Scatter(x =df["time"], y = df[c], name = c), secondary_y=False)
    # fig.update_yaxes(range=[0,vmax],secondary_y=False) #2021.07.21 for hokuriku teleme 
  for c in col2:
    fig.add_trace(go.Scatter(x =df["time"], y = df[c], name = c, fill='tozeroy'), secondary_y=True)
    # _trace.append(go.bar(x = df["time"],  y = df[c], name = c, yaxis='y2'))
  fig.update_yaxes(range=[0,vmax],secondary_y=True) #2021.07.21 for hokuriku teleme 
  # layout = go.Layout(
  #   xaxis = dict(title = 'date', type='date'),  #, dtick = 'M1'),  # dtick: 1か月ごとは'M1' 
  #   yaxis = dict(side = 'left', showgrid=False),
  #   yaxis2 = dict(side = 'right', overlaying = 'y',showgrid=False)
  
  # )
  
  # fig = dict(data = _trace, layout = layout)
  # # fig = px.line(df,x='time', y=_col, title=title)
  fig.write_html(html_path)
  return 
