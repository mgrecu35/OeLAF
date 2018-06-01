import datetime

def updateNL(yy1,mm1,day1,h1,yy2,mm2,day2,h2,lat,lon,dt1,dt2,dt41,dt42,imemb):
    ls=open('namelist.input.generic','r').readlines()
    f2=open('namelist.input.c','w')
    t0=datetime.datetime(yy1,mm1,day1,h1)
    t1=datetime.datetime(yy2,mm2,day2,h2)
    t03=t0+datetime.timedelta(hours=dt1)
    t13=t1-datetime.timedelta(hours=dt2)
    t04=t0+datetime.timedelta(hours=dt41)
    t14=t1-datetime.timedelta(hours=dt42)
    dt=t1-t0
    hours=dt.days*24+dt.seconds/3600.-3
    sd='%4.4i-%2.2i-%2.2i_%2.2i:00:00'%(t0.year,t0.month,t0.day,t0.hour)
    ed='%4.4i-%2.2i-%2.2i_%2.2i:00:00'%(t1.year,t1.month,t1.day,t1.hour)
    print sd
    t02=t1-datetime.timedelta(hours=6)
    for l in ls:
        if 'run_hours' in l:
            l=' run_hours = 0, \n'
            print l
        if 'start_year' in l:
            l=' start_year  = %4.4i, %4.4i, %4.4i, %4.4i,\n'%(t0.year,t0.year,t03.year,t04.year)
            print l
        if 'start_month' in l:
            l=' start_month  =  %2.2i, %2.2i, %2.2i, %2.2i, \n'%(t0.month,t0.month,t03.month,t04.month)
            print l
        if 'start_day' in l:
            l=' start_day  =  %2.2i, %2.2i, %2.2i, %2.2i, \n'%(t0.day,t0.day,t03.day,t04.day)
            print l
        if 'start_hour' in l:
            l=' start_hour  =  %2.2i, %2.2i, %2.2i, %2.2i, \n'%(t0.hour,t0.hour,t03.hour,t04.hour)
            print l
            
        if 'end_year' in l:
            l=' end_year  = %4.4i, %4.4i, %4.4i, %4.4i,\n'%(t1.year,t1.year,t13.year,t14.year)
            print l
        if 'end_month' in l:
            l=' end_month  =  %2.2i, %2.2i, %2.2i, %2.2i, \n'%(t1.month,t1.month,t13.month,t14.month)
            print l
        if 'end_day' in l:
            l=' end_day  =  %2.2i, %2.2i, %2.2i, %2.2i, \n'%(t1.day,t1.day,t13.day,t14.day)
            print l
        if 'end_hour' in l:
            l=' end_hour  =  %2.2i, %2.2i, %2.2i, %2.2i, \n'%(t1.hour,t1.hour,t13.hour,t14.hour)
            print l
        f2.write(l)
    f2.close()

