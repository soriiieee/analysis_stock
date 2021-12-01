# -*- coding: utf-8 -*-
# title  : [     ]
# date   : 2020.05.28
# date   : 2020.09.08 change file names
# editor : sori-machi
# action : 
#---------------------------------------------------------------------------
# module import
# 
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
import numpy as np

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
# initial
#

# from getDistanceLonLat import calc_km(lon_a,lat_a,lon_b,lat_b)
def calc_km(lon_a,lat_a,lon_b,lat_b):
    ra=6378.140  # equatorial radius (km)
    rb=6356.755  # polar radius (km)

    F = (ra-rb) / ra #henpei ratio
    #radian
    rad_lat_a= np.deg2rad(lat_a)
    rad_lon_a= np.deg2rad(lon_a)
    rad_lat_b= np.deg2rad(lat_b)
    rad_lon_b= np.deg2rad(lon_b)

    #kasei
    pa=np.arctan(rb/ra*np.tan(rad_lat_a))
    pb=np.arctan(rb/ra*np.tan(rad_lat_b))

    xx=np.arccos(np.sin(pa)*np.sin(pb)+np.cos(pa)*np.cos(pb)*np.cos(rad_lon_a-rad_lon_b))
    c1=(np.sin(xx)-xx)*(np.sin(pa)+np.sin(pb))**2/np.cos(xx/2)**2
    c2=(np.sin(xx)+xx)*(np.sin(pa)-np.sin(pb))**2/np.sin(xx/2)**2
    
    dr=F/8*(c1-c2)
    rho=ra*(xx+dr)
    return np.round(rho, 1)

if __name__ == '__main__':
    #test　177.169635
    #a東京 
    lon_a = 139.763308 
    lat_a = 35.681594
    #b長野 
    lon_b = 138.198642
    lat_b = 36.649853

    r = cal_rho(lon_a,lat_a,lon_b,lat_b)
    print(r)