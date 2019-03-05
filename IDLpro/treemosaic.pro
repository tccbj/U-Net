pro treemosaic
  COMPILE_OPT IDL2
  ENVI, /RESTORE_BASE_SAVE_FILES
  ENVI_BATCH_INIT,/NO_STATUS_WINDOW
  inpath = 'E:\tree\randomdata2\test\label\'
  outpath = 'E:\tree\analyse\'
  out_name = outpath + 'tree_pre_gt.img'


  infile = FILE_SEARCH(inpath,'*.tif',COUNT = num)     ;自动寻找文件
  print,num
  use_see_through = MAKE_ARRAY(num,value=1)
  see_through_val = MAKE_ARRAY(num,value=-999)
  fids = LONARR(num)
  dims = FLTARR(5,num)

  ENVI_OPEN_FILE, infile[0], r_fid=t_fid
  ENVI_FILE_QUERY,t_fid,nb=nb,ns=t_ns,nl=t_nl,data_type=data_type,bnames=bnames
  map_info = ENVI_GET_MAP_INFO(fid=t_fid)
  out_ps=map_info.ps[0:1]

  posarr = LON64ARR(nb,num)
  pos = LINDGEN(nb)

  FOR m=0, num-1 DO BEGIN
    posarr[*,m] = pos
    ENVI_OPEN_FILE, infile[m], r_fid=tempfid
    fids[m] = tempfid
    ENVI_FILE_QUERY,tempfid,nb=nb,ns=tempns,nl=tempnl
    dims[*,m]=[-1,0, tempns-1,0, tempnl-1]
  ENDFOR

  east = -1e34
  west = 1e34
  north = -1e34
  south = 1e34
  x0 = LONARR(num)
  y0 = LONARR(num)
  UL_corners_X = DBLARR(num)
  UL_corners_Y = DBLARR(num)

  FOR m=0,num-1 DO BEGIN
    pts = [ [dims[1,m], dims[3,m]],   $ ; UL  左上
      [dims[2,m], dims[3,m]],   $ ; UR  右上
      [dims[1,m], dims[4,m]],   $ ; LL  左下
      [dims[2,m], dims[4,m]] ]    ; LR  右下
    ENVI_CONVERT_FILE_COORDINATES, fids[m], pts[0,*], pts[1,*], xmap, ymap, /TO_MAP
    UL_corners_X[m] = xmap[0]
    UL_corners_Y[m] = ymap[0]
    east  = east > MAX(xmap)
    west = west < MIN(xmap)
    north = north > MAX(ymap)
    south = south < MIN(ymap)
  ENDFOR
  xsize = east - west
  ysize = north - south

  proj = ENVI_GET_PROJECTION(fid=fids[0])
  map_info = ENVI_MAP_INFO_CREATE(proj=proj, mc=[0,0,west,north], ps=out_ps)
  temp = BYTARR(10,10)
  ENVI_ENTER_DATA, temp, map_info=map_info, /NO_REALIZE, r_fid=tmp_fid

  x0 = LONARR(num)
  y0 = LONARR(num)
  FOR m=0,num-1 DO BEGIN
    ENVI_CONVERT_FILE_COORDINATES, tmp_fid, xpix, ypix, UL_corners_X[m], UL_corners_Y[m]
    x0[m] = xpix
    y0[m] = ypix
  ENDFOR
  ENVI_FILE_MNG, id=tmp_fid, /REMOVE, /NO_WARNING

  ENVI_DOIT, 'MOSAIC_DOIT', fid=fids, pos=posarr, $
    dims=dims, out_name=out_name, xsize=xsize, $
    ysize=ysize, x0=x0, y0=y0, georef=1, $
    out_dt=data_type, pixel_size=out_ps, out_bname=bnames, $
    background=-999,use_see_through=use_see_through,$
    see_through_val=see_through_val,map_info=map_info
  ENVI_FILE_MNG, id = t_fid, /REMOVE
  FOR m=0, num-1 DO BEGIN
    ENVI_FILE_MNG, id = fids[m], /REMOVE
  ENDFOR
END
