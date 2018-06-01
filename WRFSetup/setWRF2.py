import datetime
import os
lat=48.4
lon=-124.0+0.35
sth=0
enh=18
yy1=2014
mm1=11
dd1=30
h1=sth
yy2=2016
mm2=12
dd2=15
h2=enh
st=datetime.datetime(yy1,mm1,dd1,sth)+datetime.timedelta(hours=0)
et=datetime.datetime(yy2,mm2,dd2,enh)
ndays=et.day-st.day+1
print ndays
filL=''
print ndays


#stop
from netCDF4 import Dataset
os.environ['LD_LIBRARY_PATH']='/home/grecu/netcdf-gfortran/lib'
from processWPS2 import *
import glob
from processNamelist2 import *
import numpy as np
for i in range(0,1):
    
    st1=datetime.datetime(yy1,mm1,dd1,sth)+datetime.timedelta(hours=i*24)
    print st1, yy1, mm1, dd1
   
    en1=datetime.datetime(yy1,mm1,dd1,sth)+datetime.timedelta(hours=(i+2)*24)
   
    yy11=st1.year
    mm11=st1.month
    dd11=st1.day
    fiL1=glob.glob('%4.4i%2.2i%2.2i/*grib2'%(yy11,mm11,dd11))
    yy21=en1.year
    mm21=en1.month
    dd21=en1.day
    fiL2=glob.glob('%4.4i%2.2i%2.2i/*grib2'%(yy21,mm21,dd21))

    filL=''
    for f1 in fiL1:
        filL=filL+f1+' '
    for f1 in fiL2:
        filL=filL+f1+' '
        
        
    cmd='wget -nc ftp://nomads.ncdc.noaa.gov/GFS/analysis_only/%4.4i%2.2i/%4.4i%2.2i%2.2i/gfsanl_3_%4.4i%2.2i%2.2i_*_000.grb'%(yy11,mm11,yy11,mm11,dd11,yy11,mm11,dd11)
    #os.system(cmd)
    
    cmd='wget -nc ftp://nomads.ncdc.noaa.gov/GFS/analysis_only/%4.4i%2.2i/%4.4i%2.2i%2.2i/gfsanl_3_%4.4i%2.2i%2.2i_*_000.grb'%(yy21,mm21,yy21,mm21,dd21,yy21,mm21,dd21)
    #os.system(cmd)
    en2=datetime.datetime(yy1,mm1,dd1,sth)+datetime.timedelta(hours=(i+1)*24)
    yy31=en2.year
    mm31=en2.month
    dd31=en2.day
    cmd='wget -nc ftp://nomads.ncdc.noaa.gov/GFS/analysis_only/%4.4i%2.2i/%4.4i%2.2i%2.2i/gfsanl_3_%4.4i%2.2i%2.2i_*_000.grb'%(yy31,mm31,yy31,mm31,dd31,yy31,mm31,dd31)
    #os.system(cmd)
    

    os.system('rm GRIBF*')
    filL='*grb'
    os.system('/home/grecu/WPS/link_grib.csh *1130*.grib *1201*grib *1202*grib')
    
    h11=st1.hour
    h21=en1.hour
    updateNLWPS(yy11,mm11,dd11,h11,yy21,mm21,dd21,h21,lat,lon)
    os.system('cp namelist.wps.c namelist.wps')
    os.system('/home/grecu/WPS/geogrid.exe')
    os.system('/home/grecu/WPS/ungrib.exe')
    os.system('/home/grecu/WPS/metgrid.exe')
    #stop
    dt1=0
    dt2=0
    dt41=0
    dt42=0
    imemb=0
    
    updateNL(yy11, mm11,dd11,h11,yy21,mm21,dd21,h21,lat,lon,dt1,dt2,dt41,dt42,imemb)
    os.system('cp namelist.input.c namelist.input')
    os.system('./real.exe')
    source_mask=np.loadtxt('source_mask.txt')
    ncFile=Dataset('wrfinput_d01','r+')
    source_maski=ncFile.variables['SOURCE_MASK']
    source_maski[0,:,:]=source_mask
    os.system('./wrf.exe')
    


#
#os.system('/home/grecu/WPS/geogrid.exe')
from processNameList import *
dt1=11
dt2=6
dt41=11
dt42=8
for imemb in range(0,8):
    st=datetime.datetime(yy1,mm1,dd1,sth)+datetime.timedelta(hours=imemb*6)
    yy1m=st.year
    mm1m=st.month
    dd1m=st.day
    h1m=st.hour
    os.system('mkdir membDir_Seed3_%2.2i'%imemb)
    updateNL(yy1m, mm1m,dd1m,h1m,yy2,mm2,dd2,h2,lat,lon,dt1,dt2,dt41,dt42,imemb)
    os.system('cp namelist.input.c membDir_Seed3_%2.2i/namelist.input'%imemb)
    os.system('cp namelist.input.c namelist.input')
    #
    #
