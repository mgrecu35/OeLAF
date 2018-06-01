import sys, os
from netCDF4 import Dataset
os.system('rm -f P*2009*')
files=open('filesAN.20141201','r')
import os
for f in files:
    f1=f[:-1]
    f2=f1.replace("P","S")
    print f1,f2
    os.system("cdo -select,name=TH -chname,TH,T %s %s"%(f1,f2))
    stop
    fout=f1.replace("an","P")
    os.system("/usr/local/bin/cdo -f nc -t ecmwf copy -invertlat -chname,W,OMEGA %s %s"%(f1,fout))
    os.system("ncap2   -O -s 'PS=0.01f*exp(LNSP)' %s %s"%(fout,fout))
    cmd="ncap2   -O -s 'PS=0.01f*exp(LNSP)' %s %s"%(fout,fout)
    print cmd
    f2=f1.replace("ps","tuvwq")
    fout2=f2.replace("an","P")
    os.system("/usr/local/bin/cdo -f nc -t ecmwf copy -invertlat -chname,W,OMEGA %s %s"%(f2,fout2))
    fout3=fout.replace("_ps","tmp")
    os.system("/usr/local/bin/cdo -f nc merge %s %s %s"%(fout,fout2,fout3))
    fout4=fout3.replace("tmp","")
    os.system("/usr/local/bin/cdo sellonlatbox,-180,180,-90,90 %s %s"%(fout3,fout4))
