;+
  ; :Author: TL7050
  ;-
pro addNDVI

  COMPILE_OPT IDL2
  ENVI, /RESTORE_BASE_SAVE_FILES
  ENVI_BATCH_INIT,/NO_STATUS_WINDOW

  infile = 'E:\tree\preprocess\calibration\QUAC.dat'
;  infile = 'E:\tree\preprocess\QUAC_gt_byte.dat'
;  outpath = 'E:\tree\subsetdata\x256\'
  out_name = 'E:\tree\preprocess\calibration\NDVI12523.dat'

;  print,r
  ENVI_OPEN_FILE, infile, r_fid=r_fid
  ENVI_FILE_QUERY, r_fid, ns=ns, nl=nl ,nb=nb, dims=dims, bnames=bnames, data_type=data_type
  out_proj = ENVI_GET_PROJECTION(fid=r_fid, pixel_size=out_ps)
  map_info = ENVI_GET_MAP_INFO(fid=r_fid)
  ;print,map_info
  images = MAKE_ARRAY(ns,nl,1,/float,value=0.0)
  nir = MAKE_ARRAY(ns,nl,1,/float,value=0.0)
  r = MAKE_ARRAY(ns,nl,1,/float,value=0.0)
  nir = ENVI_GET_DATA(fid=r_fid, dims=dims, pos=6)
  r = ENVI_GET_DATA(fid=r_fid, dims=dims, pos=4)
  images = (nir*1.0-r*1.0)*1.0/(nir*1.0+r*1.0+1e-6)*1.0
;  for i=0, nb-1 do begin
;    images[*,*,i] = ENVI_GET_DATA(fid=r_fid, dims=dims, pos=i)
;  endfor
;print,images[100,100,8]
;  images[*,*,nb] = (images[*,*,6]-images[*,*,4])*1.0/(images[*,*,6]+images[*,*,4]+1e-6)*1.0
;  print,images[100,100,8]
;  for i=0, nb do begin
;    band_mean = mean(images[*,*,i])
;    band_std = stddev(images[*,*,i])
;    images[*,*,i] = (images[*,*,i]-band_mean)/band_std
;  endfor
;  for i=0, nb-1 do begin
;    tmp = images[*,*,i]
;    band_max = max(tmp)*0.35
;    band_min = min(tmp)
;    tmp[where(tmp ge band_max)] = band_max
;    images[*,*,i] = (tmp-band_min)/(band_max-band_min)
;  endfor
  
  OPENW, lun, out_name,/get_lun
  WRITEU,lun, images
  CLOSE, lun & FREE_LUN, lun   ;关闭释放内存

 ; out_bnames = ['SensorZenith','SensorAzimuth','SolorZenith','SolorAzimuth','red_sur_ref','blue_sur_ref','red_real_ref','blue_real_ref']

  ;输出hdr文件
  ENVI_SETUP_HEAD,fname = out_name,$
    ns = ns, nl =nl, nb=1,$
    ;bnames=bnames, $
    interleave=0, MAP_INFO=map_info, $
    data_type = 4,$
    offset = 0, /write
END