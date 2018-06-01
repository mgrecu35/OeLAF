import numpy as np
from wrf import *
import wrf as wrf
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# draw tissot's indicatrix to show distortion.

ncFile=Dataset('wrfout_d01_2016-07-08_00:00:00','r')

from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter

rainnc=ncFile.variables['RAINNC'][:,:,:]+ncFile.variables['RAINC'][:,:,:]+ncFile.variables['SNOWNC'][:,:,:]
lon=ncFile.variables['XLONG'][0,:,:]
lat=ncFile.variables['XLAT'][0,:,:]

m = Basemap(projection='npstere',boundinglat=50,lon_0=0,resolution='l')
for ifig in range(3,-16):
    irot=58+11
    plt.figure()
    
    
    m.drawcoastlines()
    # draw parallels and meridians.
    m.drawparallels(np.arange(-80.,81.,20.))
    m.drawmeridians(np.arange(-180.,181.,20.))
    x,y=m(lon,lat)
    dr=rainnc[ifig,:,:]-rainnc[0,:,:]
    drm=np.ma.array(dr,mask=dr<0.01)
    m.pcolormesh(x,y,drm,\
                 cmap='jet',vmax=25)
    plt.colorbar()
    
    plt.savefig('stormJ%2.2i.png'%ifig)

print (rainnc[8,:,:]-rainnc[0,:,:]).sum()
ncFile=Dataset('noFlux/wrfout_d01_2016-07-08_00:00:00','r')

from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter

rainnc2=ncFile.variables['RAINNC'][:,:,:]+ncFile.variables['RAINC'][:,:,:]+ncFile.variables['SNOWNC'][:,:,:]

for i in range(8):
    plt.figure()
    i1=i+8
    dr1=rainnc[i1,:,:]-rainnc[i1-1,:,:]
    dr2=rainnc2[i1,:,:]-rainnc2[i1-1,:,:]
    plt.scatter(dr1,dr2)
    print np.mean(dr1),np.mean(dr2), np.corrcoef(dr1.flatten(),dr2.flatten())
