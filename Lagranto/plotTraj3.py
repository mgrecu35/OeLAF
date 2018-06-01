from mpl_toolkits.basemap import Basemap, cm
# requires netcdf4-python (netcdf4-python.googlecode.com)
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import glob
files=glob.glob("P2014*")
files=sorted(files)[:-2][::-1]
P=np.zeros((60,241,480))
Pi=np.zeros((61,241,480))
pL=[]
tL=[]
rhoL=[]
dzL=[]
qL=[]
import os
from readE_P import *
e_p_ECMWF=readE_P()

nc=Dataset("trajQ2.4","r")
tlon=nc.variables['lon'][:,0,0,:]
tlat=nc.variables['lat'][:,0,0,:]
p=nc.variables['p'][:,0,0,:]
q=nc.variables['Q'][:,0,0,:]

nt=tlon.shape[1]
a=np.nonzero(p[-1,:]>350)
tlon=tlon[:,a[0]]
tlat=tlat[:,a[0]]
q=q[:,a[0]]
p=p[:,a[0]]
a=np.nonzero(p[0,:]>350)
tlon=tlon[:,a[0]]
tlat=tlat[:,a[0]]
q=q[:,a[0]]
p=p[:,a[0]]
e_p=np.zeros((21,241,480),float)
c=np.zeros((21,241,480),float)
e_p_t=np.zeros((40,21,241,480),float)
ct=np.zeros((40,21,241,480),float)
for t0 in range(1,40):
    for i in range(q.shape[1]):
        k=int((p[t0,i]-50)/50.)
        x=tlon[t0,i]
        y=tlat[t0,i]
        ix=int((x+180)/0.75)
        iy=int((y+90)/0.75)
        if k>=0 and k<21:
            e_p[k,iy,ix]+=(q[t0-1,i]-q[t0,i])/1000.*50*100./9.81/1000.
            e_p_t[t0,k,iy,ix]+=(q[t0-1,i]-q[t0,i])/1000.*50*100./9.81/1000.
            c[k,iy,ix]+=1
            ct[t0,k,iy,ix]+=1
a=np.nonzero(c>0)
e_p[a]=e_p[a]/c[a]
a=np.nonzero(ct>0)
e_p_t[a]=e_p_t[a]/ct[a]
e_p=e_p_t.sum(axis=0)
cs=ct.sum(axis=1)
import pickle
pickle.dump(cs,open('cs.pklz','wb'))

import matplotlib.pyplot as plt
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

lon,lat=np.meshgrid(-180+0.75/2+np.arange(480)*0.75,-90+np.arange(241)*0.75)
grid_def = geometry.GridDefinition(lons=lon, lats=lat)
row_indices, \
    col_indices = \
                  utils.generate_nearest_neighbour_linesample_arrays(grid_def, area_def, 200000)

msg_con = image.ImageContainer(e_p.sum(axis=0), grid_def)
e_p_grid = msg_con.get_array_from_linesample(row_indices, col_indices)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 13})

m = Basemap(projection='npstere',boundinglat=20,lon_0=0,resolution='l')
plt.suptitle('Moisture Transport from Lagrangian Analysis',fontsize=14)
plt.subplot(111)

m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,20.),labels=[True,False,False,False])
m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,True,True])

xx,yy=area_def.get_lonlats()
x2,y2=m(xx,yy)
e_p2=np.ma.array(e_p_grid,mask=abs(e_p_grid)<0.0001)
m.pcolormesh(x2,y2,e_p2,cmap='RdBu_r',vmin=-0.04,vmax=0.04)
cbar=plt.colorbar()
cbar.ax.set_title('m')
plt.savefig('WT_201401_12.png')
ct=c.sum(axis=0)
at=np.nonzero(abs(e_p2)>0.0001)
plt.figure()

plt.scatter(e_p2[at],e_p_ECMWF[at])

from sklearn.cluster import Birch, MiniBatchKMeans
stop
nc=5
mbk = MiniBatchKMeans(init='k-means++', n_clusters=nc, batch_size=1000)
mbk.fit(np.array([tlon[-15,:],tlat[-15,:]]).T)

for i in range(0,nc):
    plt.figure()
    m.drawcoastlines()
    m.drawcountries()
    m.drawparallels(np.arange(-80.,81.,20.),labels=[True,False,False,False])
    m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,True,True])
    a=np.nonzero(mbk.labels_==i)
    for i0 in a[0]:
        xt,yt=m(tlon[:-15,i0],tlat[:-15,i0])
        plt.plot(xt,yt,'.',markersize=1)


stop
stop1

from scipy.ndimage.filters import gaussian_filter

from sklearn.cluster import Birch, MiniBatchKMeans
nc=10
mbk = MiniBatchKMeans(init='k-means++', n_clusters=nc, batch_size=1000)
mbk.fit(np.array([tlon[-8,:],tlat[-8,:]]).T)

for ic in range(nc):
    plt.figure()
    dqdt=np.zeros((40,20),float)
    c=np.zeros((40,20),float)
    pgrid=50+np.arange(20)*50
    a=np.nonzero(mbk.labels_==ic)
    for t0 in range(1,q.shape[0]):
        for i in a[0]:
            k=int((p[t0,i]-50)/50.)
            if k>=0 and k<20:
                dqdt[t0-1,k]+=(q[t0-1,i]-q[t0,i])/6.
                c[t0-1,k]+=1

    a0=np.nonzero(c>0)
    dqdt[a0]=dqdt[a0]/c[a0]
    dqdtf=gaussian_filter(dqdt[::-1,::-1],0.1)
    dqdtfm=np.ma.array(dqdtf,mask=abs(dqdtf)<0.01)
    #plt.pcolormesh(-dqdtfm.T,cmap='jet')
    plt.pcolormesh(dqdtfm.T,cmap='RdBu_r',vmin=-0.2,vmax=0.2)
    plt.colorbar()

for f in files[:32]:
    nc=Dataset(f,'r')
    f2=f.replace("P","S")
    print f,f2
    #os.system("cdo -select,name=TH,PS -chname,T,TH %s %s"%(f,f2))
    ps=nc.variables['PS'][0,0,:,:]
    ai=nc.variables['hyai'][:]
    bi=nc.variables['hybi'][:]
    am=nc.variables['hyam'][:]
    bm=nc.variables['hybm'][:]
    T=nc.variables['T'][0,:,:,:]
    Q=nc.variables['Q'][0,:,:,:]
    for i in range(241):
        for j in range(480):
            P[:,i,j]=am/100+bm*ps[i,j]
            Pi[:,i,j]=ai/100+bi*ps[i,j]
    rho=P/287./T*1e2
    dP=Pi[1:,:,:]-Pi[:-1,:,:]
    dz=1e2*dP/rho/9.81
    pL.append(P)
    rhoL.append(rho)
    tL.append(T)
    dzL.append(dz)
    qL.append(Q)

qL=np.array(qL)
dzL=np.array(dzL)
tL=np.array(tL)
pL=np.array(pL)


#m = Basemap(width=9000000,height=9000000,
#            rsphere=(6378137.00,6356752.3142),\
#            resolution='l',area_thresh=1000.,projection='lcc',\
#            lat_1=50.,lat_2=60,lat_0=55,lon_0=-20.)
m = Basemap(projection='npstere',boundinglat=20,lon_0=0,resolution='l')


1
plt.figure()
plt.hist(p[-1,:])

t0=60
qL1=[]
for t0 in range(0,q.shape[0],6):
    for i in range(q.shape[1]):
        x=tlon[t0,i]
        y=tlat[t0,i]
        pz=p[t0,i]
        ix=int((x+180)/0.75)
        iy=int((y+90)/0.75)
        f1=(x+180)/0.75-ix
        f2=(y+90)/0.75-iy
        q1=qL[t0/6,:,iy,ix]*(1-f1)*(1-f2)+qL[t0/6,:,iy+1,ix]*f2*(1-f1)+\
            qL[t0/6,:,iy,(ix+1)%480]*f1*(1-f2)+qL[t0/6,:,iy+1,(ix+1)%480]*f2*f1
        p1=pL[t0/6,:,iy,ix]*(1-f1)*(1-f2)+pL[t0/6,:,iy+1,ix]*f2*(1-f1)+\
            pL[t0/6,:,iy,(ix+1)%480]*f1*(1-f2)+pL[t0/6,:,iy+1,(ix+1)%480]*f2*f1
        if pz>300:
            qint=np.interp(pz,p1,q1)
            qL1.append((qint,q[t0,i]))
    
stop
from sklearn.cluster import Birch, MiniBatchKMeans
nc=10
mbk = MiniBatchKMeans(init='k-means++', n_clusters=nc, batch_size=1000)
mbk.fit(np.array([tlon[-48,:],tlat[-48,:]]).T)

xt,yt=m(tlon[0,:],tlat[0,:])


for i in range(0,nc):
    plt.figure()
    m.drawcoastlines()
    #m.drawstates()
    m.drawcountries()
    m.drawparallels(np.arange(-80.,81.,20.),labels=[True,False,False,False])
    m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,True,True])
    a=np.nonzero(mbk.labels_==i)
    for i0 in a[0]:
        xt,yt=m(tlon[:,i0],tlat[:,i0])
        plt.plot(xt,yt,'.',markersize=1)
    plt.show()
#plt.title('Backwards Traj December 2014')
#plt.savefig('backTraj.png')
