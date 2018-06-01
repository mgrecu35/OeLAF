&share
 wrf_core = 'ARW',
 max_dom = 1,
 start_date='2014-11-30_00:00:00','2014-11-30_00:00:00',
 end_date='2014-12-02_00:00:00','2014-12-02_00:00:00',
 interval_seconds = 21600,
 io_form_geogrid = 2,
 opt_output_from_geogrid_path = '/media/grecu/ExtraDrive1/polarWRF/',
 debug_level = 0,
/

&geogrid
 parent_id         = 1,
 parent_grid_ratio = 1,
 i_parent_start    = 1,
 j_parent_start    = 1,
 e_we          = 201,
 e_sn          = 201,
 geog_data_res = '10m',
 dx = 66000,
 dy = 66000,
 map_proj =  'polar',
 ref_lat   = 90,
 ref_lon   = -0.,
 truelat1  = 69.113,
 truelat2  = 90,
 stand_lon = -0.,
 stand_lon = -0.,
 geog_data_path = '/media/grecu/ExtraDrive1/geog',
 opt_geogrid_tbl_path = '/media/grecu/ExtraDrive1/polarWRF/',
/

&ungrib
 out_format = 'WPS',
 prefix = 'FILE',
/

&metgrid
 fg_name = 'FILE',
 io_form_metgrid = 2,
 opt_output_from_metgrid_path = '/media/grecu/ExtraDrive1/polarWRF/',
 opt_metgrid_tbl_path = '/media/grecu/ExtraDrive1/polarWRF/',
/

&mod_levs
 press_pa = 201300 , 200100 , 100000 ,
             95000 ,  90000 ,
             85000 ,  80000 ,
             75000 ,  70000 ,
             65000 ,  60000 ,
             55000 ,  50000 ,
             45000 ,  40000 ,
             35000 ,  30000 ,
             25000 ,  20000 ,
             15000 ,  10000 ,
              5000 ,   1000
 /


&domain_wizard
 grib_data_path = '/media/grecu/ExtraDrive1/polarWRF/grib',
 grib_vtable = 'Vtable.ECMWF',
 dwiz_name    =polarWRF
 dwiz_desc    =polarWRF
 dwiz_user_rect_x1 =-1
 dwiz_user_rect_y1 =-46
 dwiz_user_rect_x2 =722
 dwiz_user_rect_y2 =47
 dwiz_show_political =true
 dwiz_center_over_gmt =true
 dwiz_latlon_space_in_deg =10
 dwiz_latlon_linecolor =-8355712
 dwiz_map_scale_pct =5.0
 dwiz_map_vert_scrollbar_pos =0
 dwiz_map_horiz_scrollbar_pos =0
 dwiz_gridpt_dist_km =33.0
 dwiz_mpi_command =null
 dwiz_tcvitals =null
 dwiz_bigmap =Y
/
