from mpl_toolkits.basemap import Basemap, cm
# requires netcdf4-python (netcdf4-python.googlecode.com)
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import glob

e_p=np.zeros((20,241,480),float)
nc=Dataset('evap_precip201411.nc','r')
t=nc.variables['time'][:][20*4+2:30*4+2]
t0=nc.variables['time'][:][0:1]
e=nc.variables['e'][21*4+2:31*4+2,::-1,:]
p=nc.variables['tp'][21*4+2:31*4+2,::-1,:]
for i in range(1,e.shape[0],2):
    e[i,:,:]=e[i,:,:]-e[i-1,:,:]
    p[i,:,:]=p[i,:,:]-p[i-1,:,:]
import datetime
print datetime.datetime(1900,1,1)+datetime.timedelta(hours=int(t[0]))
print datetime.datetime(1900,1,1)+datetime.timedelta(hours=int(t0[0]))
print datetime.datetime(1900,1,1)+datetime.timedelta(hours=int(t[-1]))
e_p=(e+p)
import pickle
cs=pickle.load(open('cs.pklz','rb'))
cs=np.roll(cs,240,axis=2)[::-1,:,:]
a=np.nonzero(cs<5)
e_p[a]=0.
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 13})

x_size = 251
y_size = 251
description = 'Arctic EASE grid'
proj_id = 'ease_nh'
from pyresample import geometry, utils, image
area_id = 'ease_nh'
area_extent = (-7326849.0625,-7326849.0625,7326849.0625,7326849.0625)
proj_dict = {'a': '6371228.0', 'units': 'm', 'lon_0': '0', \
             'proj': 'laea', 'lat_0': '90'}
area_def = geometry.AreaDefinition(area_id, description, proj_id, \
                                   proj_dict, x_size, y_size, area_extent)
import numpy as np
lon,lat=np.meshgrid(0+np.arange(480)*0.75,-90+np.arange(241)*0.75)
lon[lon>180]-=360.
grid_def = geometry.GridDefinition(lons=lon, lats=lat)
row_indices, \
    col_indices = \
                  utils.generate_nearest_neighbour_linesample_arrays(grid_def, area_def, 200000)

msg_con = image.ImageContainer(e_p.sum(axis=0), grid_def)
e_p_grid = msg_con.get_array_from_linesample(row_indices, col_indices)

m = Basemap(projection='npstere',boundinglat=20,lon_0=0,resolution='l')

plt.suptitle('Reanalysis E-P',fontsize=14)
plt.subplot(111)

m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,20.),labels=[True,False,False,False])
m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,True,True])

xx,yy=area_def.get_lonlats()
x2,y2=m(xx,yy)
e_p2=np.ma.array(e_p_grid,mask=abs(e_p_grid)<0.0001)
m.pcolormesh(x2,y2,-e_p2,cmap='RdBu_r',vmin=-0.04, vmax=0.04)
cbar=plt.colorbar()
cbar.ax.set_title('m')
plt.savefig('E_P_20141201_cond.png')
