import datetime
import os
sth=0
enh=18
yy1=2014
mm1=12
dd1=01
h1=sth
yy2=2014
mm2=12
dd2=31
h2=enh
lat=48.4
lon=-124.0+0.35
st=datetime.datetime(yy1,mm1,dd1,sth)+datetime.timedelta(hours=0)
et=datetime.datetime(yy2,mm2,dd2,enh)
ndays=et.day-st.day+1
print ndays
filL=''
print ndays

for i in range(ndays):
    c1=st+datetime.timedelta(days=i)
    y=c1.year
    m=c1.month
    d=c1.day
    dirp='ftp://nomads.ncdc.noaa.gov/modeldata/cfsv2_analysis_pgbh/%4.4i/%4.4i%2.2i/%4.4i%2.2i%2.2i/'%(y,y,m,y,m,d)
    os.system('mkdir %4.4i%2.2i%2.2i/'%(y,m,d))
    for itime in range(4):
        name1='cdas1.t%2.2iz.pgrbh00.grib2'%(itime*6)
        pname1='%4.4i%2.2i%2.2i/'%(y,m,d)+name1
        fexists=os.path.exists(pname1)
        filL=filL+' %s'%pname1
        if fexists:
            print pname1
        else:
            cmd='wget -nc --tries=3 --read-timeout=30 '+dirp+name1
            print cmd
            os.system(cmd)
    os.system('mv cdas*grib2 %4.4i%2.2i%2.2i/'%(y,m,d))

stop
print filL
from processWPS import *
from processNameList import *
os.environ['LD_LIBRARY_PATH']='/home/grecu/netcdf-gfortran/lib'
#os.system('/home/grecu/WPS/link_grib.csh '+filL)
#updateNLWPS(yy1,mm1,dd1,h1,yy2,mm2,dd2,h2,lat,lon)
#os.system('cp namelist.wps.c namelist.wps')
#os.system('/home/grecu/WPS/geogrid.exe')
#os.system('/home/grecu/WPS/ungrib.exe')
#os.system('/home/grecu/WPS/metgrid.exe')
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
    #os.system('./real.exe')
    #os.system('./wrf.exe')
