import numpy as np
from wrf import *
import wrf as wrf
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# draw tissot's indicatrix to show distortion.

ncFile=Dataset('wrfout_d01_2014-11-30_00:00:00','r')

from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter

import wrf as wrf
from netCDF4 import Dataset
import glob
files=glob.glob("wrfout_*2014*")
files=sorted(files)
from trackP import *

uL=[]
vL=[]
wL=[]
dzL=[]
zmL=[]
pL=[]
tL=[]
for f in files[0:1]:
    nc=Dataset(f,'r')
    nt=nc.variables['Times'].shape[0]
    for i in range(nt):
        u=wrf.getvar(nc,"ua",i)
        v=wrf.getvar(nc,"va",i)
        w=wrf.getvar(nc,"wa",i)
        z=wrf.getvar(nc,"zstag",i)
        p = wrf.getvar(nc, 'pressure',i)
        t = wrf.getvar(nc, 'theta',i)
        dz=z[1:,:,:]-z[:-1,0,0]
        zm=0.5*(z[1:,:,:]+z[:-1,0,0])
        uL.append(u.data)
        vL.append(v.data)
        wL.append(w.data)
        pL.append(p.data)
        dzL.append(dz.data)
        zmL.append(zm.data)
        tL.append(t.data)
        
dx=66000
dy=66000
mapf=nc.variables["MAPFAC_M"][0,:,:]
qv=nc.variables["QVAPOR"][:,:,:,:]
lon=nc.variables["XLONG"][0,:,:]
lat=nc.variables["XLAT"][0,:,:]

from mpl_toolkits.basemap import Basemap

m = Basemap(projection='npstere',boundinglat=30,lon_0=0,resolution='l')
plt.figure()
xx,yy=m(lon[40:51,30:41],lat[40:51,30:41])
m.drawcoastlines()
plt.scatter(xx,yy)

import numpy as np
u=np.array(uL)
v=np.array(vL)
w=np.array(wL)
dz=np.array(dzL)
zm=np.array(zmL)
p=np.array(pL)
t=np.array(tL)
dt=180.
tinc=3600
ns=30
ix=[]
iy=[]
iz=[]
rainnc=ncFile.variables['RAINNC'][:,:,:]+ncFile.variables['RAINC'][:,:,:]
dr=rainnc[-1,:,:]-rainnc[-2,:,:]

for i in np.linspace(81,101,60):
    for j in np.linspace(75,96,60):
        for k in np.linspace(3,30,50):
            if dr[int(j),int(i)]>0.1:
                ix.append(i)
                iy.append(j)
                iz.append(k)


ix=np.array(ix)
iz=np.array(iz)
iy=np.array(iy)

print zm.shape
reverse=1
if reverse==0:
    pospartout,lonOut,latOut,hout = trace3dball(u,v,w,dz,zm,mapf,\
                                                lon,lat,ix,\
                                                iy,iz,dx,dy,\
                                                dt,tinc)
else:
    u1=-u[::-1,:,:,:]
    v1=-v[::-1,:,:,:]
    w1=-w[::-1,:,:,:]
    dz1=dz[::-1,:,:,:]
    zm1=zm[::-1,:,:,:]
    qv1=qv[::-1,:,:,:]
    p1=p[::-1,:,:,:]
    t1=t[::-1,:,:,:]
    rainnc1=rainnc[::-1,:,:]
    print 'here'
    pospartout,lonOut,latOut,hout = trace3dball(u1,v1,w1,dz1,zm1,mapf,\
                                                lon,lat,ix ,\
                                                iy ,iz ,dx,dy,\
                                                dt,tinc)
    qvL=[]
    ppL=[]
    wL=[]
    tL=[]
    rL=[]
    for i in range(latOut.shape[0]):
        ix=pospartout[i,:,0]
        iy=pospartout[i,:,1]
        iz=pospartout[i,:,2]
        qvpart = getpartprop(qv1[i,:,:,:],ix,iy,iz)
        ppart = getpartprop(p1[i,:,:,:],ix,iy,iz)
        tpart = getpartprop(t1[i,:,:,:],ix,iy,iz)
        wpart = getpartprop(w1[i,:,:,:],ix,iy,iz)
        rpart = getpartprop2(rainnc1[i,:,:],ix,iy,iz)
        
        #print qvpart.min()
        qvL.append(qvpart)
        ppL.append(ppart)
        tL.append(tpart)
        wL.append(wpart)
        rL.append(rpart)

qvL=np.array(qvL)
ppL=np.array(ppL)
tL=np.array(tL)
wL=np.array(wL)
rL=np.array(rL)

e_p=np.zeros((41,200,200),float)

rain1=rainnc[::-1,:,:]

dbL=[]
a1=np.nonzero(pospartout[-1,:,-1]>0)
ind=np.argmax(pospartout[0,:,-1][a1]-pospartout[-1,:,-1][a1])
isort=np.argsort(pospartout[0,:,-1][a1]-pospartout[-1,:,-1][a1])[-100:]
plt.figure()
plt.plot(qvL[::-1,a1[0][isort]].mean(axis=1))
qvPlot=qvL[::-1,a1[0][isort[-100:]]].mean(axis=1)
e_c=(qvL[::-1,ind][1:]-qvL[::-1,ind][:-1])
dbLver=[]
c=np.zeros((41,200,200),float)
e_p_t=np.zeros((48,41,200,200),float)
ct=np.zeros((48,41,200,200),float)
for t0 in range(1,48):
    for i in range(qvL.shape[1]):
        k=int((ppL[t0,i]-50)/25.)
        x=pospartout[t0,i,0]
        y=pospartout[t0,i,1]
        ix=int(x)
        iy=int(y)
        x1=pospartout[t0-1,i,0]
        y1=pospartout[t0-1,i,1]
        ix1=int(x1)
        iy1=int(y1)
        if k>=0 and k<41:
            if ix*(ix-199)<=0 and iy*(iy-199)<=0 and \
               qvL[t0,i]>=0 and qvL[t0-1,i]>=0:
                e_p[k,iy,ix]+=(qvL[t0-1,i]-qvL[t0,i])*25*100./9.81/dt
                e_p_t[t0,k,iy,ix]+=(qvL[t0-1,i]-qvL[t0,i])*25*100./9.81/dt
                c[k,iy,ix]+=1
                ct[t0,k,iy,ix]+=1
                sfrain=rain1[t0-1,ix,iy]-rain1[t0,ix,iy]
                sfrain1=rain1[t0-1,ix1,iy1]-rain1[t0,ix,iy]
                sfrain=rL[t0-1,i]-rL[t0,i]
                if(ppL[t0,i]>50):
                    dbL.append([sfrain,sfrain1,ppL[t0,i],tL[t0,i],wL[t0-1,i],wL[t0,i],qvL[t0,i]*1000.,\
                                (qvL[t0-1,i]-qvL[t0,i])*1000.])
                    if i in a1[0][isort]:
                        dbLver.append([sfrain,sfrain1,ppL[t0,i],tL[t0,i],wL[t0-1,i],wL[t0,i],qvL[t0,i]*1000.,\
                                       (qvL[t0-1,i]-qvL[t0,i])*1000.,t0])
                    

qTSL=[]
qTSLver=[]

for i in range(qvL.shape[1]):
    if pospartout[:,i,:].min()>0:
        qt=[qvL[0,i]*1000]
        sfcRainL=[]
        p1L=[]
        t1L=[]
        for t0 in range(1,48):
            k=int((ppL[t0,i]-50)/25.)
            x=pospartout[t0,i,0]
            y=pospartout[t0,i,1]
            ix=int((x))
            iy=int((y))
            if k>=0 and k<41:
                if ix*(ix-199)<=0 and iy*(iy-199)<=0 and \
                   qvL[t0,i]>0 and qvL[t0-1,i]>0:
                    sfrain=(rain1[t0-1,ix,iy]-rain1[t0,ix,iy])
                    sfcRainL.append(sfrain)
                    p1L.append(ppL[t0-1,i])
                    t1L.append(tL[t0-1,i])
                    qt.append(qvL[t0,i]*1000)
        qTSL.append([qt,sfcRainL,p1L,t1L])
        if i==ind:
            qTSLver.append([qt,sfcRainL,p1L,t1L])

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
#sfrain,sfrain1,ppL[t0-1,i],tL[t0-1,i],wL[t0-1,i],wL[t0,i],qvL[t0,i]
varL=[0,1,2,4,5]
#regressor=KNeighborsRegressor(n_neighbors=50,weights='distance')
dbL=np.array(dbL)
#a0=np.nonzero(dbL[:,0]>0.0000001)
#dbL=dbL[a0[0],:]
dbLver=np.array(dbLver)
r=np.random.random(dbL.shape[0])
a=np.nonzero(r<0.5)
b=np.nonzero(r>0.5)
varL=[0,1,2,3,4,5,6]
regressor.fit(dbL[a[0],:][:,varL], dbL[a[0],-1])
#c=np.nonzero(dbL[b[0],0]>0.00001)
y=regressor.predict(dbL[b[0],:][:,varL])
yver=regressor.predict(dbLver[:,varL])
print np.corrcoef(y,dbL[b[0],-1])
print np.corrcoef(yver,dbLver[:,-2])

plt.figure()
s=(qvPlot[1:]-qvPlot[:-1])*1000.
for i in range(1,47):
    s[i]+=s[i-1]


s2=yver.reshape(47,100).mean(axis=1)[::-1]
yv=yver.reshape(47,100).mean(axis=1)[::-1]
for i in range(1,47):
    s2[i]+=s2[i-1]

    
matplotlib.rcParams.update({'font.size': 14})
plt.figure(figsize=(9,7))
fig=plt.subplot(312)
plt.plot(np.arange(47)+0.5,(qvPlot[1:]-qvPlot[:-1])*1000.,linewidth=2)
plt.plot(np.arange(47)+0.5,yv*s[-1]/s2[-1],linewidth=2)
fig.axes.get_xaxis().set_visible(False)
plt.ylabel('g/kg')
plt.legend(['Truth','Estimated'])
plt.title('Hourly $e-c$')
plt.subplot(313)
plt.plot(np.arange(47)+0.5,s,linewidth=2)
plt.plot(np.arange(47)+0.5,s2*s[-1]/s2[-1],linewidth=2)
plt.title('Path-integrated $e-c$')
plt.xlabel('Time (h)')
plt.ylabel('g/kg')
plt.legend(['Truth','Estimated'])
fig=plt.subplot(311)
plt.title('Specific humidity $q_v$')
plt.plot(np.arange(48),1000*qvPlot,linewidth=2)
plt.ylabel('g/kg')
fig.axes.get_xaxis().set_visible(False)
plt.savefig('qvTSeries2.png')

#for i in range(47):
    
pickle.dump([y,dbL[b[0],4]], open('wrfDBase0.pklz'))
stop
a=np.nonzero(c>0)
e_p[a]=e_p[a]/c[a]
matplotlib.rcParams.update({'font.size': 12})
plt.figure()
m = Basemap(projection='npstere',boundinglat=50,lon_0=0,resolution='l')
plt.subplot(111)
plt.suptitle('E-P from Lagrangian analysis\n 2 Dec 2014, 00:00:00 UTC',x=0.45)
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,20.),labels=[True,False,False,False])
m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True])
xx,yy=m(lon,lat)
e_p2=np.ma.array(e_p.sum(axis=0),mask=abs(e_p.sum(axis=0))<0.00001)
plt.pcolormesh(xx,yy,e_p2*3600,cmap='RdBu_r',vmax=10, vmin=-10)
cbar=plt.colorbar()
cbar.ax.set_title('mm/h')
plt.savefig('WRF_E_P_20141202_00.png')
stop
plt.figure()
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,20.),labels=[True,False,False,False])
m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True])
plt.subplot(111)
plt.suptitle('WRF 1 hour precipitation\n 2 Dec 2014, 00:00:00 UTC',x=0.45)
xx,yy=m(lon,lat)
drm=np.ma.array(dr,mask=dr<0.01)
plt.pcolormesh(xx,yy,drm,vmax=4,cmap='jet')
cbar=plt.colorbar()
cbar.ax.set_title('mm/h')
a=np.nonzero(dr[75:96,80:101]>0.1)
xx,yy=m(lon[75:96,80:101][a],lat[75:96,80:101][a])
plt.scatter(xx,yy,s=0.3,color='red')
plt.savefig('wrf_precip_20141202_00.png')

a=np.nonzero(lonOut[-1,:]>-998)
ic=0
for i in range(lonOut.shape[1]):
    if hout[:,i].min()>0:
        if hout[0,i]-hout[-1,i]>6000:
            x1,y1=m(lonOut[:,i],latOut[:,i])
            plt.scatter(x1,y1,s=0.05,color='red')
            ic+=1
