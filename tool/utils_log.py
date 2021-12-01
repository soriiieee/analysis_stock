# -*- coding: utf-8 -*-
# when   : 2020.0x.xx
# who : [sori-machi]
# what : [ ]
#---------------------------------------------------------------------------
# basic-module
if 1:
  import os, sys, gc
  import glob
  from datetime import datetime


def log_write(path,mes,init=False):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    mes2 = f"{now} {mes}\n" #改行コードの追加
    if init:
        with open(path,"w") as f:
            f.write(mes2)
    else:
        with open(path,"+a") as f:
            f.write(mes2)
    return 