;+
  ; :Author: TL7050
  ;-
pro subset

  COMPILE_OPT IDL2
  ENVI, /RESTORE_BASE_SAVE_FILES
  ENVI_BATCH_INIT,/NO_STATUS_WINDOW

  infile = 'E:\tree\preprocess\NND_strech_float.dat'
;  infile = 'E:\tree\preprocess\QUAC_gt_byte.dat'
;  outpath = 'E:\tree\subsetdata\x256\'
  outpath = 'E:\tree\raw_data\x8\'
  e=envi(/headless)
  r = e.OpenRaster(infile)
;  print,r
  ENVI_OPEN_FILE, infile, r_fid=r_fid
  ENVI_FILE_QUERY, r_fid, ns=ns, nl=nl ,nb=nb, dims=dims, bnames=bnames, data_type=data_type
;  out_proj = ENVI_GET_PROJECTION(fid=r_fid, pixel_size=out_ps)
;  map_info = ENVI_GET_MAP_INFO(fid=r_fid)
;  images = MAKE_ARRAY(300,300,nb,/float,value=0.0)
;  ns=300
;  nl=300
  i = 0
  stride = 256
  for x=0,ns-1,stride do begin
    for y=0,nl-1,stride do begin
      ;set start and end of x
      if x+stride-1 gt ns-1 then begin
        tmpx_start = ns-1-stride+1
        tmpx_end = ns-1
      endif else begin
        tmpx_start = x
        tmpx_end = x+stride-1
      endelse
      ;set start and end of y
      if y+stride-1 gt nl-1 then begin
        tmpy_start = nl-1-stride+1
        tmpy_end = nl-1
      endif else begin
        tmpy_start = y
        tmpy_end = y+stride-1
      endelse
;      input_dims = [-1L, tmpx_start, tmpx_end, tmpy_start, tmpy_end]
;      for i=0, nb-1 do begin
;        images[*,*,i] = ENVI_GET_DATA(fid=r_fid, dims=input_dims, pos=i)
;      endfor
      outfile = outpath + strtrim(string(i),2) + '.tif'
;      outfile = outpath + strtrim(string(i+1681),2) + '.tif'
      subset_image =  ENVISubsetRaster(r, SUB_RECT=[tmpx_start, tmpy_start, tmpx_end, tmpy_end])
      subset_image.Export, outfile,'TIFF'
      i = i+1
    endfor
  endfor
  
  
;return,images
;  for i=0, nb-1 do begin
;    images[i,*,*] = ENVI_GET_DATA(fid=r_fid, dims=dims, pos=i)
;  endfor
;  print,size(images)
;  band1 = ENVI_GET_DATA(fid=r_fid, dims=dims, pos=0)
;  band2 = ENVI_GET_DATA(fid=r_fid, dims=dims, pos=1)
;  band3 = ENVI_GET_DATA(fid=r_fid, dims=dims, pos=2)
;  band4 = ENVI_GET_DATA(fid=r_fid, dims=dims, pos=3)
;  band5 = ENVI_GET_DATA(fid=r_fid, dims=dims, pos=4)
;  band6 = ENVI_GET_DATA(fid=r_fid, dims=dims, pos=5)
;  band7 = ENVI_GET_DATA(fid=r_fid, dims=dims, pos=6)
;  band8 = ENVI_GET_DATA(fid=r_fid, dims=dims, pos=7)
END
