# -*- coding: utf-8 -*-
# title  : [     ]
# date   : 2020.0x.xx
# editor : sori-machi
# action : 
#---------------------------------------------------------------------------
# module import
# 
import urllib.request
import sys
import os
import pprint
import time
import urllib.error

sys.path.append('/home/griduser/tool')
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
# data get

def download(url,datPath):
    try:
        with urllib.request.urlopen(url) as web_file:
            #data_ file instance
            data = web_file.read()
            with open(datPath, mode='wb') as local_file:
                #writing mode
                local_file.write(data)
    except urllib.error.URLError as e:
        print(e, url)
    return

def dl_file(url, dst_dir):
    download(url, os.path.join(dst_dir, os.path.basename(url)))
    return


# if __name__ == "__main__":
    
    #sample est
