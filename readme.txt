
FOR SEGMENTATION ::

Robust statistics segmenter: based on active contours
  Works great for swine_IVC (pixel spacing there is 0.12x0.12x1mm)
  Does not work always for phantom images - probably because initial segmentation inputLabelMap region is too small. In that case the outputLabelMap is actually a shrinkning. 
  What helps is : either decrease the pixel spacing or make the initial inputLabelMap bigger. 
  Do not forget to increase the expected approximate volume (esp with pixel spacing 1x1x1)
  PRO: circular shape can be defined by boundary smoothness.  
  
  WORKS when the pixel spacing is small. or better yet - the initial segment is big enough.
  Things might run more smoothly after smoothVolume because then random bright speckles in blood will be discarded. 
  where smoothVol = labelmap(threshold (>100) + keep_largerst_island) >> Did not work - dont understand why
  OR origVolMaskedBy(threshold (>100) + keep_largerst_island) >> Did not work
  OR MakeInsideHollowBY(remove_really_tiny_islands) -> works (manually)
   
Simple Region Growing Segmentation: 
  SO SLOW but also works using a single fiducial seed too.
  outputLabelMap will have a bee-hive type of appearance which can be controlled by smoothing A LOT. or by doing minor preprocessing 
  CON: Prone to leaking. 
  
 
Canny Edge Detection: 
  quite sensitive to all kinds of edges. Tested on Phantom3_tracked image 
  initial estimates: keep variance a bit high (high gaussian blur) like 4 to avoid subtle changes. 
  initial estimate: keep both thresholding as 10. better yet lower =0, upper =10 
  
ANOTHER APPROACH: instead of treating one slice/image at a time, load teh segmentation as a 3D volume and work with that. 
  - Later get each slice as image and convert to 2.5D or BETTER: convert the labelmap to a new 3D volume. (apply tfm to ones only)
  To get the basic framework: threshold (>100) + keep_largerst_island + joint_smoothing.
  
 To try: 
  vessel segmentation module 
  MATLab Active contour 
  IVUS segmentation by other people (on desktop, maltalb)
  
  
MATLAB ACTIVE CONTOURS: Works, but better when small islands are removed first. or maybe just threshold all below 90 or 100. 