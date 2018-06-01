#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer
server = ECMWFDataServer()
import datetime
st=datetime.datetime(2014,11,21,12)

for i in range(28,11*4-1):
    ct=st+datetime.timedelta(hours=i*6)
    print ct
    y=ct.year
    m=ct.month
    d=ct.day
    h=ct.hour
    server.retrieve({
        "class": "ei",
        "dataset": "interim",
        "date": "%4.4i-%2.2i-%2.2i/to/%4.4i-%2.2i-%2.2i"%(y,m,d,y,m,d),
        "expver": "1",
        "grid": "0.75/0.75",
        "levelist": "1/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24/25/26/27/28/29/30/31/32/33/34/35/36/37/38/39/40/41/42/43/44/45/46/47/48/49/50/51/52/53/54/55/56/57/58/59/60",
        "levtype": "ml",
        "param": "130.128/131.128/132.128/133.128/135.128",
        "step": "0",
        "stream": "oper",
        "time": "%2.2i:00:00"%h,
        "type": "an",
        "target": "an%4.4i%2.2i%2.2i_%2.2i_tuvwq"%(y,m,d,h),
    })
    
    server.retrieve({
        "class": "ei",
        "dataset": "interim",
        "date": "%4.4i-%2.2i-%2.2i/to/%4.4i-%2.2i-%2.2i"%(y,m,d,y,m,d),
        "expver": "1",
        "grid": "0.75/0.75",
        "levelist": "1",
        "levtype": "ml",
        "param": "152.128",
        "step": "0",
        "stream": "oper",
        "time": "%2.2i:00:00"%h,
        "type": "an",
        "target": "an%4.4i%2.2i%2.2i_%2.2i_ps"%(y,m,d,h),
    })
