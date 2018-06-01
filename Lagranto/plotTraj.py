from mpl_toolkits.basemap import Basemap, cm
# requires netcdf4-python (netcdf4-python.googlecode.com)
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col

import matplotlib.pyplot as plt
m = Basemap(width=9000000,height=9000000,
            rsphere=(6378137.00,6356752.3142),\
            resolution='l',area_thresh=1000.,projection='lcc',\
            lat_1=50.,lat_2=60,lat_0=55,lon_0=-20.)
m.drawcoastlines()
m.drawstates()
m.drawcountries()
m.drawparallels(np.arange(-80.,81.,20.),labels=[True,False,False,False])
m.drawmeridians(np.arange(-80.,81.,20.),labels=[False,False,True,True])

nc=Dataset("wcb.4","r")
tlon=nc.variables['lon'][:,0,0,:]
tlat=nc.variables['lat'][:,0,0,:]
nt=tlon.shape[1]
for i in range(nt):
    xt,yt=m(tlon[:,i],tlat[:,i])
    plt.plot(xt,yt)

plt.title('WCB identification using Lagranto\nSprenger and Wenli (2015)')
plt.savefig('wcb.png')
