from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import glob
import datetime
from pyresample import geometry, utils, image
def readE_P():
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
    e_p=np.zeros((20,241,480),float)
    nc=Dataset('evap_precip201411.nc','r')
    t=nc.variables['time'][:][20*4+2:30*4+2]
    t0=nc.variables['time'][:][0:1]
    e=nc.variables['e'][21*4+2:31*4+2,::-1,:]
    p=nc.variables['tp'][21*4+2:31*4+2,::-1,:]
    for i in range(1,e.shape[0],2):
        e[i,:,:]=e[i,:,:]-e[i-1,:,:]
        p[i,:,:]=p[i,:,:]-p[i-1,:,:]
        
    print datetime.datetime(1900,1,1)+datetime.timedelta(hours=int(t[0]))
    print datetime.datetime(1900,1,1)+datetime.timedelta(hours=int(t0[0]))
    print datetime.datetime(1900,1,1)+datetime.timedelta(hours=int(t[-1]))
    e_p=(e+p)
    lon,lat=np.meshgrid(0+np.arange(480)*0.75,-90+np.arange(241)*0.75)
    lon[lon>180]-=360.
    grid_def = geometry.GridDefinition(lons=lon, lats=lat)
    row_indices, \
        col_indices = \
                      utils.generate_nearest_neighbour_linesample_arrays(grid_def, area_def, 200000)
    
    msg_con = image.ImageContainer(e_p.sum(axis=0), grid_def)
    e_p_grid = msg_con.get_array_from_linesample(row_indices, col_indices)
    return e_p_grid
