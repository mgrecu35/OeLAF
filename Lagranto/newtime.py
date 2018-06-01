#!/usr/bin/env python
import datetime
import sys
start_time=sys.argv[1]
timeinc=sys.argv[2]
#print sys.argv[2]
s1=datetime.datetime(int(start_time[0:4]),int(start_time[4:6]),int(start_time[6:8]),int(start_time[9:11]))
s2=s1+datetime.timedelta(hours=int(timeinc))
#print s1
#print s2
s2s='%4.4i%2.2i%2.2i_%2.2i'%(s2.year,s2.month,s2.day,s2.hour)
#print s1
#print s2
print s2s
#stop
