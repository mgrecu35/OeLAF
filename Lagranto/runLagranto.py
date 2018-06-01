import os
os.environ['LAGRANTO']='/home/grecu/lagranto/lagranto.ecmwf'
os.environ['LD_LIBRARY_PATH']='/home/grecu/netcdf-gfortran/lib'

cmd='../../bin/create_startf 20141201_12 startf_201401_12 "circle.eqd(-10,75,3000,35) @ profile(50,800,40) @hPa,agl"'
os.system(cmd)
#stop
cmd='../../bin/caltra 20141201_12 20141121_12 startf_201401_12 traj2.4  -t 15 -o 360 -j'
os.system(cmd)
cmd='../../bin/trace traj2.4 trajQ2.4 -f Q 1000. 0 P'
os.system(cmd)

