# -*- coding: utf-8 -*-
# when   : 2020.0x.xx
# who : [sori-machi]
# what : [ ]
#---------------------------------------------------------------------------
# basic-module
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from matplotlib.colors import ListedColormap, BoundaryNorm, rgb2hex

import sys,os,re,glob
# import pandas as pd
import numpy as np
#---------------------------------------------------
# sori -module
sys.path.append('/home/ysorimachi/tool')
import subprocess
# outd ='/home/griduser/work/sori-py2/deep/out/0924_01'
# os.makedirs(outd, exist_ok=True)
# subprocess.run('rm -f *.png', cwd=outd, shell=True)
f_name = sys.argv[0].split(".")[0]

def cmap_from_list(cols,sepa=False):
  """
  2021.03.02 入力した配列で色がグラデーションで出てくる
  ['r','yellow','g','b','magenta']
  """
  nmax = float(len(cols)-1)
  _color = [] #tuple list
  
  for i,c in enumerate(cols):
    _color.append(tuple([i/nmax, c]))
  
  if sepa:
    return mpl.colors.ListedColormap(cols)
  else:
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', _color)

def user_cmap(_val,_color,sepa=True):
  """
  reference...
  input
    _val : list(value)
    _color: list("color" ex "k"=block)
  return
    cmap, norm (for imshow)
  
  #カスタムカラー
  http://hydro.iis.u-tokyo.ac.jp/~akira/page/Python/contents/plot/color/colormap.html
  #色指定について
  https://qiita.com/github-nakasho/items/1d5209992b00a31b8eae
  
  """
  if sepa:
    cmap = mpl.colors.ListedColormap(_color)
  else:
    cmap = cmap_from_list(_color,sepa=False)
  cmap.set_under('silver')
  cmap.set_over('silver')
  norm = mpl.colors.BoundaryNorm(_val, cmap.N)
  
  return cmap, norm


def cmap_snow():
  """
  北陸技研用
  user_cmapから積雪に使える気象庁準拠のcmapとnormを取得を取得
  """
  _color =["lightblue","cornflowerblue","b","yellow","orange","tomato","m"]
  _val = [0,5,20,50,100,150,200,500] 
  cmap,norm = user_cmap(_val,_color,sepa=True)
  return cmap,norm
  
def cmap_smame(_val,name="jet"):
  """
  北陸技研用
  user_cmapから積雪に使える気象庁準拠のcmapとnormを取得を取得
  """
  N = len(_val)
  def get_color_code(name,num):
    """
    sub routine
    """
    cmap = cm.get_cmap(name,num)
    code_list =[]
    for i in range(cmap.N):
      rgb = cmap(i)[:3]
      # print(rgb2hex(rgb))
      code_list.append(rgb2hex(rgb))
    return code_list
  #main
  color_code = get_color_code(name,N)
  cmap = ListedColormap(color_code)
  norm = BoundaryNorm(_val,cmap.N)
  return cmap,norm

def main():
  """
  カラーバーの数値を独自に作成する
  http://hydro.iis.u-tokyo.ac.jp/~akira/page/Python/contents/plot/color/colormap.html
  """
  
  a = np.random.rand(100).reshape(10,10)
  a = a*150
  # print(np.max(a), np.min(a))
  # sys.exit()
  f,ax = plt.subplots(figsize=(10,5))
  _color =["lightblue","cornflowerblue","b","yellow","orange","tomato","m"]
  _val = [0,5,20,50,100,150,200,500] 
  cmap,norm = user_cmap(_val,_color,sepa=True)
  
  
  # print(cmap(np.arange(5)))
  # sys.exit()
  # cf = ax.imshow(a, cmap=plt.cm.jet, interpolation='nearest')
  
  vmin, vmax = np.min(_val), np.max(_val)
  # print(vmin, vmax )
  cf = ax.imshow(a,cmap=cmap,norm=norm, interpolation='nearest',vmin=vmin, vmax=vmax )
  plt.colorbar(cf,extend='both',shrink=0.7)
# cf1 = ax[1].imshow(a, cmap=plt.cm.jet)
# plt.colorbar(cf1)
  plt.savefig(f"./{f_name}.png", bbox_inches="tight")

# print(plt.cm.jet(np.arange(256)))

if __name__ == "__main__":
  main()