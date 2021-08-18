import os
import unittest
import vtk, qt, ctk, slicer
from qt import QSlider
from slicer.ScriptedLoadableModule import *
import logging
import slicer.util
import math
import numpy as np
import time
import sitkUtils
import SimpleITK as sitk
try: 
  import cv2
except ImportError:
  slicer.util.pip_install('opencv-python')
  import cv2 

# 
# vessel reconstruction
#

class VascularPhantomReconstruction(ScriptedLoadableModule):
  """
  Basic description 
  """
  
  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Vascular Phantom Reconstruction"
    self.parent.categories = ["US vessel reconstruction"]
    self.parent.contributors = ["Hareem Nisar & Leah Groves (VASST Lab, Western Uni)"] 
    self.parent.helpText = """ Vessel reconstruction from tracked (Conavi's) ICE images"""
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """ VASST LAB """

#
# test1Widget
#

class VascularPhantomReconstructionWidget(ScriptedLoadableModuleWidget):
  """
  GUI control 
  """
  def __init__(self, parent=None):
    ScriptedLoadableModuleWidget.__init__(self, parent)
    slicer.mymod = self
    self.logic = VascularPhantomReconstructionLogic()
    self.sequenceBrowserLogic = slicer.modules.sequencebrowser.logic()
  
  def setup(self):
    # this is the function that implements all GUI
    ScriptedLoadableModuleWidget.setup(self)
    self.logic = VascularPhantomReconstructionLogic()
    
    #member variables
    
    #the cosmetic 5
    self.imgSeqNode = None
    self.inLMSeqNode = None
    self.outLMSeqNode = None
    self.out3DLMSeqNode = None
    self.tfmSeqNode = None
    self.imToP = None
    self.im = None
    self.probe = None
    self.CA_mask = None 
    self.calTrans = None
    self.inputLabelMap = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode', 'InputLabelMap')
    self.outputLabelMap = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode', 'OutputLabelMap')
    self.tempVol = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode', "Temporary Volume")
    
    self.imgCorr0 = np.eye(4) #to fix origin
    self.imgCorr0[0,3] = -424
    self.imgCorr0[1,3] = -424
    
    self.segLog = slicer.modules.segmentations.logic()
    self.MHN_CA = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelHierarchyNode')
    self.APD_CA = vtk.vtkAppendPolyData()
    self.BMF = sitk.BinaryMorphologicalClosingImageFilter()
    self.BMF.SetKernelType(self.BMF.Ball)
    self.BMF.SetKernelRadius([30,30,30])
    self.GF = sitk.SmoothingRecursiveGaussianImageFilter()
    self.GF.SetSigma([0.5, 0.5,0.5])
    self.resliceLogic = slicer.modules.volumereslicedriver.logic()
    self.WL = sitk.IntensityWindowingImageFilter() 
    self.WL.SetWindowMinimum(10)
    self.WL.SetWindowMaximum(100)
    self.transformNode = None 
    self.imageNode = None    
    self.NN = 0
    # Instantiate and connect widgets ...
    
    #
    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "2.5D Reconstruction"
    self.layout.addWidget(parametersCollapsibleButton)
    
    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)
    
    #
    # input Image sequence selector
    #
    self.imgSeqSelector = slicer.qMRMLNodeComboBox()
    self.imgSeqSelector.nodeTypes = ["vtkMRMLSequenceNode"]
    self.imgSeqSelector.selectNodeUponCreation = True
    self.imgSeqSelector.addEnabled = True
    self.imgSeqSelector.removeEnabled = True
    self.imgSeqSelector.noneEnabled = True
    self.imgSeqSelector.showHidden = False
    self.imgSeqSelector.showChildNodeTypes = False
    self.imgSeqSelector.setMRMLScene( slicer.mrmlScene )
    self.imgSeqSelector.setToolTip( "Select the cropped image sequence for segmentation" )
    parametersFormLayout.addRow("Image Sequence: ", self.imgSeqSelector)
    
    #
    # input Transform sequence selector
    #
    self.tfmSeqSelector = slicer.qMRMLNodeComboBox()
    self.tfmSeqSelector.nodeTypes = ["vtkMRMLSequenceNode"]
    self.tfmSeqSelector.selectNodeUponCreation = True
    self.tfmSeqSelector.addEnabled = True
    self.tfmSeqSelector.removeEnabled = True
    self.tfmSeqSelector.noneEnabled = True
    self.tfmSeqSelector.showHidden = False
    self.tfmSeqSelector.showChildNodeTypes = False
    self.tfmSeqSelector.setMRMLScene( slicer.mrmlScene )
    self.tfmSeqSelector.setToolTip( "Select the probe transform sequence" )
    parametersFormLayout.addRow("Probe Transform Sequence: ", self.tfmSeqSelector)
    
    #
    # calib Transform  selector
    #
    self.tfmSelector = slicer.qMRMLNodeComboBox()
    self.tfmSelector.nodeTypes = ["vtkMRMLLinearTransformNode"]
    self.tfmSelector.selectNodeUponCreation = True
    self.tfmSelector.noneEnabled = True
    self.tfmSelector.showHidden = False
    self.tfmSelector.showChildNodeTypes = False
    self.tfmSelector.setMRMLScene( slicer.mrmlScene )
    self.tfmSelector.setToolTip( "Select the calibration transform" )
    parametersFormLayout.addRow("Calibration Transform: ", self.tfmSelector)
    
    #numOfSamples
    self.numOfSamples = ctk.ctkSliderWidget()
    self.numOfSamples.decimals = 0
    self.numOfSamples.singleStep = 1
    self.numOfSamples.minimum = 0
    self.numOfSamples.maximum = 194
    self.numOfSamples.value =  20
    self.numOfSamples.setToolTip("Choose numOfSamples for reconstruction")
    parametersFormLayout.addRow("Number of samples to reconstruct vessel:",self.numOfSamples)
    
    #angle
    self.angle = ctk.ctkSliderWidget()
    self.angle.decimals = 0
    self.angle.singleStep = 1
    self.angle.minimum = 30
    self.angle.maximum = 89
    self.angle.value =  67
    self.angle.setToolTip("Choose imaging angle")
    parametersFormLayout.addRow("Angle phi (in degrees):",self.angle)
    
    #
    # Apply Segmentation + Tfm Button
    #
    self.applySegTfmButton = qt.QPushButton("2D segmentations + cones + transforms")
    self.applySegTfmButton.toolTip = "Get segmentation of vessel from the seq, make cones, and apply transforms"
    self.applySegTfmButton.enabled = True
    parametersFormLayout.addRow(self.applySegTfmButton)
    
    #
    # Apply Smoothing only Button
    #
    self.applySmoothingButton = qt.QPushButton("Apply smoothing")
    self.applySmoothingButton.toolTip = "Get segmentation of vessel from the seq"
    self.applySmoothingButton.enabled = False
    parametersFormLayout.addRow(self.applySmoothingButton)
    
    # connections
    self.applySegTfmButton.connect('clicked(bool)', self.onApplySegTfmButton)
    self.applySmoothingButton.connect('clicked(bool)', self.onSmooth)
   
    # Add vertical spacer
    self.layout.addStretch(1)
    
    self.imgSeqSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.onImageChanged)
    self.tfmSeqSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.onTransformChanged)
    self.tfmSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.onCalibrationSelected)
    
    #set observer for auto-update
    self.imgObserverTag = None
    print("in setup, oberverTag value:  ", self.imgObserverTag)
    
    # Refresh Apply button state
    #self.onSelect()
  def onImageChanged(self):
    if self.imageNode is not None: 
      self.image = None 
    self.imageNode = self.imgSeqSelector.currentNode() 
    if self.imageNode is None:
      print('Please select an Image Sequence')
    else:
      self.seq = self.imgSeqSelector.currentNode()
      self.numOfSamples.maximum = self.seq.GetNumberOfDataNodes()
      
  def onTransformChanged(self):
    if self.transformNode is not None: 
      self.transformNode = None 
    self.transformNode = self.tfmSeqSelector.currentNode() 
    if self.transformNode is None:
      print('Please select a Probe Sequence')
  
  def onCalibrationSelected(self):
    if self.tfmSelector.currentNode() is not None: 
      self.calibMatrix = self.tfmSelector.currentNode()
      self.imToP = slicer.util.arrayFromTransformMatrix(self.calibMatrix)
      print("inside onCalibrationSelected")
    
  #end def
  
  def cleanup(self):
    slicer.mrmlScene.RemoveNode(self.inputLabelMap)
    slicer.mrmlScene.RemoveNode(self.outputLabelMap)
    slicer.mrmlScene.RemoveNode(self.tempVol)
    print("in clean up")
    if self.imgObserverTag is not None:
      print("before removing observer. tag value: ", self.imgObserverTag)
      self.inputSelector.currentNode().RemoveObserver(self.imgObserverTag)
      print("after removing observer. tag value: ", self.imgObserverTag)
    
  def onSelect(self):
    self.applyButton.enabled = self.inputSelector.currentNode() and self.outputSelector.currentNode()
    if self.imgObserverTag is None and self.inputSelector.currentNode() is not None:
      self.imgObserverTag = self.inputSelector.currentNode().AddObserver(slicer.vtkMRMLScalarVolumeNode().DisplayModifiedEvent, self.onModification)
      print("adding observer. tag value: ", self.imgObserverTag)
    #end if
  
  def onGenerateBSButton(self):
    logic = VascularPhantomReconstructionLogic()
    logic.generateBackground(self.outputSelector.currentNode(), self.AngleSlider.value)
      
  def onApplyBSButton(self):
    logic = VascularPhantomReconstructionLogic()
    logic.subtractBackground(self.imgSeqSelector.currentNode(), self.threshold.value)
  def Mat4x4ToArray(self, mat):
    array = np.zeros([4,4])
    for i in range(0,4):
      for j in range(0,4):
        array[i,j] = mat.GetElement(i,j)
    return array     
  
  def onApplySegTfmButton(self): # final version with segmentation visualization and vessel reconstruction
        	
    start_time = time.time()
    
    ## Adding a new Seq Browser Node to add all label maps to
    self.sequenceBrowserNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSequenceBrowserNode', 'New SeqBrowser')
    #self.sequenceBrowserNode.SetAttribute("Recorded", "True")
    self.sequenceBrowserNode.SetScene(slicer.mrmlScene)
    slicer.mrmlScene.AddNode(self.sequenceBrowserNode)
    
    # creating sequences to store data processes
    self.imgSeqNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSequenceNode', 'Image Sequence')
    self.inLMSeqNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSequenceNode', 'Input LabelMap Sequence')
    self.outLMSeqNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSequenceNode', 'Output LabelMap Sequence')
    self.out3DLMSeqNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSequenceNode', 'Output 3D LabelMap Sequence')
    self.tfmSeqNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSequenceNode', 'Transform Select Sequence') 
    
    #generating an empty input/seed labelmap and output labelmap
    self.inputLabel = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
    self.outputLabel = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
    self.inputLabel.SetName('tempInLabelMap')
    self.outputLabel.SetName('tempOutLabelMap')
    
    mat = vtk.vtkMatrix4x4()
    Tr = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTransformNode')
    CAseg = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
        
    self.tfmSeq = self.tfmSeqSelector.currentNode()
    self.seq = self.imgSeqSelector.currentNode()
    
    max  = self.seq.GetNumberOfDataNodes()-10
    samples = np.int(self.numOfSamples.value)
    sp = np.int(np.round(max/samples))
    for i in range(0, max, sp):
      
      #getting input volume for segmentation
      self.inputVolNode  = self.seq.GetNthDataNode(i)
      self.seqIndex      = self.seq.GetNthIndexValue(i)
      #print(self.inputVolNode.GetName())
      self.tempVol.Copy(self.inputVolNode)
      
      # Prepare transform Tr (calib matrix and probe position combined
      self.P = [self.tfmSeq.GetNthDataNode(i)] # sequence files
      dum = self.P[0].GetMatrixTransformToWorld(mat)
      A = self.Mat4x4ToArray(mat)
      self.probe = [A]
      self.imToP_Corr = np.matmul(self.imToP, self.imgCorr0)
      self.calTrans =[np.matmul(self.probe[0], self.imToP_Corr)]
      slicer.util.updateTransformMatrixFromArray(Tr, self.calTrans[0])
      
      # CA will be the segmentation output and converted to 3D later
      CA = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
      self.outLMSeqNode.SetDataNodeAtValue(CA,  self.seqIndex)
      
      # get segmentation as labelmap 
      self.logic.getLumenSegmentation(self.tempVol, self.inputLabel, CA)
      
      # and then do 3D reconstruction of the segmented Label map
      anglePhi = np.int(self.angle.value)
      CA = self.logic.reconstruction_VOL(CA, CA, anglePhi)
      
      #LEAH : make sure the origin is set to zero as assumed in calibration. No origin shifting due to cropping
      CA.SetOrigin(0,0,0)
      
      CA.SetAndObserveTransformNodeID(Tr.GetID())
      CA.HardenTransform() ## 
      
      print(i)
      
      self.segLog.ImportLabelmapToSegmentationNode(CA,CAseg) ##empty CAseg
      CAseg.CreateClosedSurfaceRepresentation()
      seg_CA = CAseg.GetSegmentation().GetSegment(CA.GetName())
      PD_CA = seg_CA.GetRepresentation('Closed surface')
      self.APD_CA.AddInputData(PD_CA)
      
      #Push labelmaps to  sequences for storage and viewing 
      self.imgSeqNode.SetDataNodeAtValue(  self.tempVol, self.seqIndex)
      self.inLMSeqNode.SetDataNodeAtValue( self.inputLabel,   self.seqIndex)
      self.out3DLMSeqNode.SetDataNodeAtValue(CA,  self.seqIndex)
      self.tfmSeqNode.SetDataNodeAtValue(self.P[0], self.seqIndex) #or Tr
      
      slicer.mrmlScene.RemoveNode(CA)
      
    #end for
    
    self.APD_CA.Update()
    self.out_CA = self.APD_CA.GetOutput() #returns polydata
    self.model_CA = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode')
    self.model_CA.SetAndObservePolyData(self.out_CA)
    self.model_CA.SetName('CA_model')
    
    slicer.mrmlScene.RemoveNode(CAseg)
    slicer.mrmlScene.RemoveNode(Tr)
    
    #adding the sequences to the 'new sequence browser' to view simultaneously
    self.sequenceBrowserNode.AddProxyNode(self.imgSeqNode.GetNthDataNode(0),  self.imgSeqNode)
    self.sequenceBrowserNode.AddProxyNode(self.inLMSeqNode.GetNthDataNode(0), self.inLMSeqNode)
    self.sequenceBrowserNode.AddProxyNode(self.outLMSeqNode.GetNthDataNode(0),self.outLMSeqNode)
    self.sequenceBrowserNode.AddProxyNode(self.out3DLMSeqNode.GetNthDataNode(0),self.out3DLMSeqNode)
    self.sequenceBrowserNode.AddProxyNode(self.tfmSeqNode.GetNthDataNode(0),    self.tfmSeqNode)
    
    print("---done with RECONSTRUCTION  only---#samples: ", samples )
    print("--- %s time to segment (minutes) ---" ,  (time.time() - start_time)/60)
    
    self.applySmoothingButton.enabled = True
  
  def onSmooth(self):
    self.segNodeCA= slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
    self.LMNodeCA= slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
    self.LMNodeCA.SetName('CA Label Map')
    self.segLog.ImportModelToSegmentationNode(self.model_CA, self.segNodeCA)
    self.segLog.ExportAllSegmentsToLabelmapNode(self.segNodeCA, self.LMNodeCA)
    V_CA = sitkUtils.PullVolumeFromSlicer(self.LMNodeCA)
    V_BMF = self.BMF.Execute(V_CA)
    V_CA = self.GF.Execute(V_CA)
    sitkUtils.PushVolumeToSlicer(V_BMF, self.LMNodeCA)
    #slicer.mrmlScene.RemoveNode(self.model_CA)
    #slicer.mrmlScene.RemoveNode(self.segNodeCA)
       
    self.segNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
    self.segNode.SetName('Vessels')
    self.segLog.ImportLabelmapToSegmentationNode(self.LMNodeCA, self.segNode)
    self.segNode.GetSegmentation().GetSegment('CA Label Map').SetColor([1,0,0])
    self.segNode.CreateClosedSurfaceRepresentation()
    print("  ** * * * smoothing is complete  * * * ** ")
      
   
#
# VascularPhantomReconstruction Logic
#

class VascularPhantomReconstructionLogic(ScriptedLoadableModuleLogic):
  """Functions come here
  """
  def __init__(self, parent=None):
    self.temp = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode', 'TempVol for HoughTfm')
    self.img2 = None
    self.prev_cPoint = None
    self.GMF = sitk.GrayscaleMorphologicalClosingImageFilter()
    self.GMF.SetKernelType(1) ##self.GMF.Ball
    self.GMF.SetKernelRadius([50,50,1])
    
    self.CF = sitk.CurvatureFlowImageFilter()
    self.CF.SetTimeStep(0.5)
    self.CF.SetNumberOfIterations(5)
    
    #self.MF = sitk.MedianImageFilter()
    #self.MF.SetRadius((3,3,1))
  
  
  def getLumenSegmentation(self, inputVolume, label, outLabel):
    #fiducial3DNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", 'myFiducials3D')
    
    ## Remove central artefact data from the image
    inputVolume = self.removeCenter_Circle(inputVolume, 13) #node, radius
    ### Get the centroid of central vein using Hough Transform
    cPoint = self.getCenterByHoughTransform(inputVolume)
    
    if cPoint is not False:
      cy, cx, radius = cPoint[0,0,:]
      self.prev_cPoint = cPoint
    else:
      cy, cx, radius = self.prev_cPoint[0,0,:]
      #cx, cy = self.getCentroidByThreshold(inputVolume, 120) #node, threshold (90orig)
    
    ##fiducial3DNode.AddFiducial(cx', cy', 0) #inRAS coods
    
    label.CopyOrientation(inputVolume)
    
    # because garbage in garbage out for RSS
    # !! gotta shrink the radius until there is no overlap
    ## !! make sure that the input values at the segment bubble are not > 100.
    bubble = self.createBubble(inputVolume, cx, cy, radius-5)#vol, cx, cy, rad
    
    slicer.util.updateVolumeFromArray(label, np.expand_dims(np.short(bubble), axis=0))
    
    # using curve flow filter to remove edges because of em tracker
    sitkVol = sitkUtils.PullVolumeFromSlicer(inputVolume)
    sitkVol2 = self.CF.Execute(sitkVol)
    sitkUtils.PushVolumeToSlicer(sitkVol2, inputVolume)
    
    ## RSS 
    # # #getting list of parameters for CLI modules. [see 'run CLI module via python console']
    parameters = {}
    parameters["expectedVolume"] = 6          # (Approximate volume(mL)) 1mL = 1cubicMillimeter/1000 (original 7.7)
    parameters["intensityHomogeneity"] = 0.8    # (Intensity Homogeneity[0-1.0])(original 0.8)
    parameters["curvatureWeight"] = 1           # (Boundary Smoothness[0-1.0])(original 1)
    parameters["labelValue"] =  1               # (Output Label Value)
    parameters["originalImageFileName"] = inputVolume   # (Original Image)
    parameters["labelImageFileName"]    = label         # (Label Image)
    parameters["segmentedImageFileName"]= outLabel      # (Output Volume)

    cliNode = slicer.cli.runSync(slicer.modules.robuststatisticssegmenter, None, parameters)
    
    #slicer.mrmlScene.RemoveNode(label)
    #reset the origin of outLabel to be 0,0
    
    ##push this output labelmap to a sequence.
    return outLabel

  def createBubble(self, volnode, cx, cy, radius):
    #make no changes to volnode - only shrink bubble based on it 
    img = slicer.util.arrayFromVolume(volnode)
    img = np.squeeze(img)
    
    shape = volnode.GetImageData().GetDimensions()
    bubble = self.circle_mask(shape[0:2],[cx, cy],radius)
    
    #shrink the bubble if it overlaps with the vessel reflections (increment 5 pixel)
    maxOverlapPixel = (np.max(img[bubble==1]))
    thresh = 115
    #print("maximum value pixel overlapping with bubble= ", maxOverlapPixel)
    while (maxOverlapPixel>thresh):
      radius = radius-5
      bubble = self.circle_mask(shape[0:2],[cx, cy],radius)
      maxOverlapPixel = (np.max(img[bubble==1]))
      #print("maximum value pixel overlapping with bubble= ", maxOverlapPixel)
    #final bubble values:
    #print("final bubble value:", cx, cy, radius)
    return bubble
  #end def
    
  def circle_mask(self, shape,centre,radius):
    """
    Return a boolean mask of 'shape' - with a ones circle of radius r at point 'center'.
    """
    x,y = np.ogrid[:shape[0],:shape[1]]
    cx,cy = centre
    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    # circular mask
    circmask = r2 <= radius*radius
    #circmask = circmask.astype(int)
    return circmask
  #end def

  def removeCenter_Circle(self, volnode, radius):
    img = slicer.util.arrayFromVolume(volnode)
    img = np.squeeze(img)
    Row, Col = img.shape
    origin = ((int(Row/2), int(Col/2)))
      
    mask = self.circle_mask(img.shape,origin,radius)
    img[mask]=0
    
    slicer.util.updateVolumeFromArray(volnode, np.expand_dims(img, axis=0))
    return volnode
  #end def

  def getCentroidByThreshold(self, inputVolume, thresh):
    ## getting origin because of shift in orgin due to cropping 
    rx, ry, rz = inputVolume.GetOrigin()
    ## get the voxel data from the image
    img = slicer.util.arrayFromVolume(inputVolume)
    img = np.squeeze(img)
    
    ## Considering vessel within a smaller portion in the middle. ignoring side-vessels and surrounding tissue.
    rad = 126 ## RadiusOfInterest
    Row, Col = img.shape
    ox, oy = ((int(Row/2), int(Col/2)))
    
    ## threshold
    thresh_img = np.copy(img, 'C') 
    thresh_img[thresh_img<thresh] = 0
    thresh_img[thresh_img>thresh] = 1
    
    ## find centroid of non-zero pixel values
    points = np.where(thresh_img==1) #print(np.shape(points))
    ## ALSO adjusting the centroid size because of cropping image earlier
    centroid = (np.mean(points[0])  , np.mean(points[1]))
    #print("Before adjustment", centroid)
    
    origin = ((ox, oy))
    edge   = ((centroid[0], centroid[1]))
    dist = math.sqrt(sum( (edge - origin)**2 for edge, origin in zip(edge, origin)))
    #print(dist)
    if dist>55:    #i.e. the point identified is more than expected radius(+10) away from the center, then pull the point inwards
      w = 45.0/dist #weight for the point to be moved
      centroid = ( (w*np.mean(points[0])+(1-w)*oy) , (w*np.mean(points[1])+(1-w)*ox) )
      #print(w, origin, edge )
      #print()
      #print("After adjustment", centroid)
    
    #cent =[] #list
    ## find centroid of non-zero pixel values
    # points = np.where(thresh_img==1) #print(np.shape(points))
    # centroid = ((np.mean(points[1]), np.mean(points[0]))) #centroid is a list here
    # print(centroid)
    # cent.append(centroid)
    return centroid

  def getCenterByHoughTransform(self, volnode):
    
    #self.temp = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode', 'TempVol for HoughTfm')
    self.temp.CopyOrientation(volnode)
    #self.temp.SetName('HT_'+volnode.GetName())
    
    ## get the voxel data from the image and update values in a tem volume
    img = slicer.util.arrayFromVolume(volnode)
    img = np.squeeze(img)
    # to detect less number of circles, blur more
    img = cv2.medianBlur(img, 15) #15 for orig datatset
    slicer.util.updateVolumeFromArray(self.temp, np.expand_dims(img, axis=0))
   
    #apply morphological closing
    sitkVol = sitkUtils.PullVolumeFromSlicer(self.temp)
    sitkVol2 = self.GMF.Execute(sitkVol)
    sitkUtils.PushVolumeToSlicer(sitkVol2, self.temp)
    
    self.img2 = slicer.util.arrayFromVolume(self.temp)
    self.img2 = np.squeeze(self.img2)
    # https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=houghcircles
    ##(InputArray img, OutputArray circles, int method, double dp, double minDist(80), double param1=100, double param2=100, int minRadius=0, int maxRadius=0 )
    circles = cv2.HoughCircles(self.img2,cv2.HOUGH_GRADIENT,1,100,param1=95,param2=20,minRadius= 50, maxRadius= 80)
    
    #slicer.mrmlScene.RemoveNode(self.temp)
    
    if np.shape(circles)==():
      print("*****Hough transform failed ")
      #print(volnode.GetName())
      return False
    else:
      #! ! make sure the self.temp image value at circles =0 && in case of multiple circles pick the one closest to the center of the image. 
      
      circles = np.uint16(np.around(circles)) #(x,y,radius).N
      print(circles)
      return circles #[cy, cx, radius]
    #end if
  #end def  
  
  
  def reconstruction_VOL(self, inputVolume, outputVolume, anglePhi):#reconstruction by vectorization only (no rendering)
    
    #logging.info('Processing started')
    #start_time = time.time()
    phi = float(anglePhi)
    
    imgArr = slicer.util.arrayFromVolume(inputVolume)
    imgArr = np.squeeze(imgArr)
    Row, Col = imgArr.shape
    oRow, oCol =  int(Row/2), int(Col/2)
    #print("size of the image is", Row, Col)
    
    #Defining the origin/apex of the cone as the middle point of the image. 
    origin = ((oRow, oCol))
    edge   = ((Row, Col))
    
    #Calculating the maximum value of z based on the max displacement of the image from the center 
    # i.e from the middle of the image (cone apex) to the corner of the image.
    max_d = math.sqrt(sum( (edge - origin)**2 for edge, origin in zip(edge, origin)))
    max_z = max_d*math.tan(math.radians(90-phi))
    
    #making an empty volume (choose an appropriate value of z)
    vol = np.zeros((int(max_z+1), Row, Col), dtype=np.uint16)
    
    #reconstruction: keeping x-axis and y-axis as same, 
    #calculation a height for each pixel value and populating volume 
    #origin stays where it is, peripherals moved along z-axis.
    result = np.where(imgArr) #get index of all voxels, for label map replace imgArr==1
    X = result[0] #row
    Y = result[1] #col
    D = np.sqrt((X-origin[0])**2 + (Y-origin[1])**2) # 6.6 us for X= np.zeros(200) 
    Z = D*math.tan(math.radians(90-phi))
    Z = Z.astype(int)
    vol[Z,X,Y] = imgArr[X,Y] 
    
    #print("--- %s time to reconstruct (seconds) ---" % (time.time() - start_time))
    
    # update the volume(data) in the volume node with new array 
    slicer.util.updateVolumeFromArray(outputVolume, vol)
    
    return outputVolume
   