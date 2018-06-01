#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer
server = ECMWFDataServer()
server.retrieve({
    "class": "ei",
    "dataset": "interim",
    "date": "2014-11-01/to/2014-12-02",
    "expver": "1",
    "grid": "0.75/0.75",
    "levtype": "sfc",
    "param": "136.128/142.128/143.128/182.128/228.128",
    "step": "6/12",
    "stream": "oper",
    "time": "00:00:00/12:00:00",
    "type": "fc",
    "format": "netcdf",
    "target": "evap_precip201411.nc",
})
