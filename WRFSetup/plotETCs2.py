import numpy as np
from wrf import *
import wrf as wrf
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# draw tissot's indicatrix to show distortion.

ncFile=Dataset('wrfout_d01_2014-12-01_00:00:00','r')

from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter

rainnc=ncFile.variables['RAINNC'][:,:,:]+ncFile.variables['RAINC'][:,:,:]+1*ncFile.variables['SNOWNC'][:,:,:]

rainnc=ncFile.variables['tr_Atlantic'][:,:,:,:]
qfx=ncFile.variables['QFX'][:,:,:]


lon=ncFile.variables['XLONG'][0,:,:]
lat=ncFile.variables['XLAT'][0,:,:]
sea_ice=ncFile.variables['SEAICE'][:,:,:]
m = Basemap(projection='npstere',boundinglat=30,lon_0=0,resolution='l')
plt.figure()


m.drawcoastlines()
# draw parallels and meridians.
m.drawparallels(np.arange(-80.,81.,20.))
m.drawmeridians(np.arange(-180.,181.,20.))
x,y=m(lon,lat)
m.pcolormesh(x,y,sea_ice[0,:,:],\
             cmap='jet')#,vmin=250,vmax=280)
plt.colorbar()
nt=rainnc.shape[0]
for ifig in range(nt-1,nt):
    irot=58+10

    plt.figure()
    m.drawcoastlines()
    # draw parallels and meridians.
    m.drawparallels(np.arange(-80.,81.,20.))
    m.drawmeridians(np.arange(-180.,181.,20.))
    x,y=m(lon,lat)
    dr=rainnc[-1,0,:,:]
    drm=np.ma.array(dr,mask=dr<0.0001)
    m.pcolormesh(x,y,drm,\
                 cmap='jet')#,norm=matplotlib.colors.LogNorm())
    plt.colorbar()
    
    plt.savefig('stormJ_2_%2.2i.png'%ifig)
