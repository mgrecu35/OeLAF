import numpy as np
#from wrf import *
#import wrf as wrf
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import geopandas as gp
seas=gp.read_file('/home/grecu/tutorial_contents-master/vector/data/oceans.shp')
print seas.head

#!1-Atlantic polygon(6), 2-Pacific polygon(1), 3-Land, 4-Undef, 5-Arctic polygon(3)

#for in range(7):
# draw tissot' indicatrix to show distortion.

ncFile=Dataset('wrfout_d01_2014-12-01_00:00:00','r')

from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter

rainnc=ncFile.variables['RAINNC'][:,:,:]+ncFile.variables['RAINC'][:,:,:]+1*ncFile.variables['SNOWNC'][:,:,:]
lon=ncFile.variables['XLONG'][0,:,:]
lat=ncFile.variables['XLAT'][0,:,:]
landmask=ncFile.variables['LANDMASK'][0,:,:]
source_mask=landmask.copy()
sea_ice=ncFile.variables['SEAICE'][:,:,:]
ny,nx=lon.shape
from shapely.geometry import Point
for i in range(nx):
    for j in range(ny):
        p1=Point(lon[j,i],lat[j,i])
        if p1.within(seas.geometry[6]):
            source_mask[j,i]=1
        if p1.within(seas.geometry[1]):
            source_mask[j,i]=2
        #if p1.within(seas.geometry[3]):
        #    source_mask[j,i]=5
source_mask[landmask==1]=3
source_mask[sea_ice[0,:,:]==1]=5
source_mask[0,:]=4
source_mask[-1,:]=4
source_mask[:,0]=4
source_mask[:,-1]=4
m = Basemap(projection='npstere',boundinglat=30,lon_0=0,resolution='l')
x,y=m(lon,lat)
m.pcolormesh(x,y,source_mask)
np.savetxt('source_mask.txt',source_mask)
plt.colorbar()
stop

m = Basemap(projection='npstere',boundinglat=30,lon_0=0,resolution='l')
plt.figure()

def averageF(q):
    avgQ = (q[:,0:-1,:,:] +q[:,1:,:,:]) / 2.0
    return avgQ

it=0
qv=ncFile.variables['QVAPOR'][it:,:,:,:]
u=ncFile.variables['U'][it:,:,:,:]
v=ncFile.variables['V'][it:,:,:,:]
rain=ncFile.variables['RAINNC'][it:,:,:]+ncFile.variables['RAINC'][it:,:,:] # mm
qfx=ncFile.variables['QFX'][it:,:,:] # kg m-2 s-1 #upward moisture flux
p=ncFile.variables['PB'][it:,:,:,:]+ncFile.variables['P'][it:,:,:,:]
m=ncFile.variables['MAPFAC_M'][it:,:,:]

stop
qav=averageF(qv)
uav=averageF(u)
vav=averageF(v)
dx = 66000.
dy = 66000.
delp=p[:,1:,:,:]-p[:,0:-1,:,:]
delp=-delp
nt,ny,nx=m.shape
tpw=(qav*delp).sum(axis=1)/(9.81)
from numpy import *
divQ=zeros((nt,ny,nx),float)
nh=3
for i in range(1,nx-1):
    for j in range(1,ny-1):
        qFlx_x2=((qav[:,:,j,i+1]+qav[:,:,j,i])*0.5*\
                 uav[:,:,j,i+1]*delp[:,:,j,i]).sum(axis=1)/9.81
        qFlx_x1=((qav[:,:,j,i]+qav[:,:,j,i-1])*0.5*\
                 uav[:,:,j,i]*delp[:,:,j,i]).sum(axis=1)/9.81
        qFlx_y2=((qav[:,:,j+1,i]+qav[:,:,j,i])*0.5*\
                 vav[:,:,j+1,i]*delp[:,:,j,i]).sum(axis=1)/9.81
        qFlx_y1=((qav[:,:,j-1,i]+qav[:,:,j,i])*0.5*\
                 vav[:,:,j,i]*delp[:,:,j,i]).sum(axis=1)/9.81
        divQ[:,j,i]=m[0,j,i]*(qFlx_x2-qFlx_x1)/dx+m[0,j,i]*(qFlx_y2-qFlx_y1)/dy
        
plt.contourf(tpw[-1,1:-1,1:-1]-tpw[0,1:-1,1:-1]+\
             divQ[:,1:-1,1:-1].mean(axis=0)*nt*nh*3600)
lhs=tpw[-1,1:-1,1:-1]-tpw[0,1:-1,1:-1]+divQ[:,1:-1,1:-1].mean(axis=0)*nt*nh*3600
rhs=(rain[-1,1:-1,1:-1]-rain[0,1:-1,1:-1])-qfx[:,1:-1,1:-1].mean(axis=0)*nt*nh*3600
print corrcoef(lhs.flatten(),rhs.flatten())
plt.figure()
plt.subplot(211)
plt.contourf(lhs)
plt.subplot(212)
plt.contourf(-rhs)
#dt(Qm)+m*(dx(u Qm)+dy(u Qm))=0
#U=mud u/m
#V=mud v/m
# Qm=mud Q
