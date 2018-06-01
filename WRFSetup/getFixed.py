from ecmwfapi import ECMWFDataServer
server = ECMWFDataServer()
server.retrieve({
"class": "ei",
"dataset": "interim",
"date": "1989-01-01",
"expver": "1",
"grid": "0.75/0.75",
"levtype": "sfc",
"param": "129.128",
"step": "0",
"stream": "oper",
"target": "Geopotential.grb",
"time": "12",
"type": "an",
})


#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer
server = ECMWFDataServer()
server.retrieve({
 "class": "ei",
 "dataset": "interim",
 "date": "1989-01-01",
 "expver": "1",
 "grid": "0.75/0.75",
 "levtype": "sfc",
 "param": "172.128",
 "step": "0",
 "stream": "oper",
 "target": "LSM.grb",
 "time": "12",
 "type": "an",
})

