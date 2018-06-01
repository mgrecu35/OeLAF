import datetime

def updateNLWPS(yy1,mm1,day1,h1,yy2,mm2,day2,h2,lat,lon):
    ls=open('namelist.wps.generic','r').readlines()
    f2=open('namelist.wps.c','w')
    t0=datetime.datetime(yy1,mm1,day1,h1)
    t1=datetime.datetime(yy2,mm2,day2,h2)
    sd='%4.4i-%2.2i-%2.2i_%2.2i:00:00'%(t0.year,t0.month,t0.day,t0.hour)
    ed='%4.4i-%2.2i-%2.2i_%2.2i:00:00'%(t1.year,t1.month,t1.day,t1.hour)
    print sd
    for l in ls:
        if 'ref_lat' in l:
            l=' ref_lat  = %7.3f \n'%lat
            print l
        if 'ref_lon' in l:
            l=' ref_lon  = %7.3f \n'%lon
            print l
        if 'truelat1' in l:
            l=' truelat1  = %7.3f \n'%lat
            print l
        if 'truelat2' in l:
            l=' truelat2  = %7.3f \n'%lat
            print l
        if 'stand_lon' in l:
            l=' stand_lon  = %7.3f \n'%lon
            print l
        if 'start_date' in l:
            ls=l.split()
            l0=" "+ls[0]+ls[1]+"'"+sd+"','"+sd+"',\n"
            l=l0
        if 'end_date' in l:
            ls=l.split()
            l0=" "+ls[0]+ls[1]+"'"+ed+"','"+ed+"',\n"
            l=l0
        if ('ref_x' not in l) and ('ref_y' not in l):
            f2.write(l)
