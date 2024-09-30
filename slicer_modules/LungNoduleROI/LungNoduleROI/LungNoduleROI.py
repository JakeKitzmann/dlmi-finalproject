import logging
import os

import vtk

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

import numpy as np
import SimpleITK as sitk
import sitkUtils



class LungNoduleROI(ScriptedLoadableModule):

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Lung Nodule ROI"  
        self.parent.categories = ["Deep Learning Lung Nodule Segmentation"] 
        self.parent.dependencies = [] 
        self.parent.contributors = ["Jake Kitzmann (Advanced Pulmonary Physiomic Imaging Laboratory -- Carver College of Medicine)"]
        self.parent.helpText = ""
        self.parent.acknowledgementText = ""

class LungNoduleROIWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False
        self.nodeList = []
        self.currentVolume = None

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/LungNoduleROI.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = LungNoduleROILogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        


        # Buttons
        # self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)
        self.ui.noduleCentroidButton.connect('clicked(bool)', self.onNoduleCentroidButton)
        self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)

        self.ui.roiSizeSlider.minimum = 4
        self.ui.roiSizeSlider.maximum = 35
        self.ui.roiSizeSlider.value = 10
        self.ui.roiSizeSlider.connect('valueChanged(int)', self.onRoiSliderValueChanged)

        self.ui.roiSizeLabel.setText(f'{self.ui.roiSizeSlider.value * 2} Slices Cubed')

        # Combobox
        self.ui.volumeComboBox.setMRMLScene(slicer.mrmlScene)
        self.ui.volumeComboBox.connect('currentNodeChanged(vtkMRMLNode*)', self.onVolumeSelected)
        self.ui.volumeComboBox.renameEnabled = True

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

        # variables for roi creation

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.GetNodeReference("InputVolume"):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

    def onNoduleCentroidButton(self):
        fiducialNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode')
        fiducialNode.SetName('nodule_centroid')

        slicer.modules.markups.logic().StartPlaceMode(0)


    def onRoiSliderValueChanged(self):
        self.ui.roiSizeLabel.setText(f'{self.ui.roiSizeSlider.value * 2} Slices Cubed')


    def onApplyButton(self):

        volume = self.ui.volumeComboBox.currentNode()
        node = slicer.mrmlScene.GetFirstNodeByName('nodule_centroid')

        if volume is None:
            logging.error('No volume selected')
            return

        # Get the centroid of the nodule
        if node is None:
            logging.error('No centroid selected')
            return
        
        # Get the centroid of the nodule
        centroid = [0, 0, 0]
        node.GetNthFiducialPosition(0, centroid)
        print(f'centroid: {centroid}')

        # Get the origin and spacing of the volume
        origin = volume.GetOrigin()
        print(f'origin: {origin}')
        spacing = volume.GetSpacing()
        print(f'spacing: {spacing}')

        difference_vector = np.subtract(centroid, origin)
        print(f'difference xyz: {difference_vector}')

        difference_slices = np.divide(difference_vector, spacing)
        difference_slices_int = []

        # convert to int and take absolute value
        for i in range(3):
            slice = int(difference_slices[i])
            if slice < 0:
                slice = -1 * slice
            difference_slices_int.append(slice)

        print(f'difference in slices: {difference_slices_int}')

        self.ui.centroidLabel.setText(difference_slices_int)

        # run image ROI class
        print()
        print('Creating image ROI...')
        roi_img = self.create_roi(volume, difference_slices_int, self.ui.roiSizeSlider.value)

    def create_roi(self, volume, centroid, size):
        print(f'Creating ROI with size {size * 2} and centroid {centroid}')

        sitk_img = sitkUtils.PullVolumeFromSlicer(volume.GetID())
        imageROI = ImageROI()
        roi_img_np = imageROI.create_roi_image(sitk_img, size, centroid)

        roi_img_volume = slicer.util.addVolumeFromArray(roi_img_np)
        roi_img_volume.SetName(self.ui.fileName.text)
        slicer.util.setSliceViewerLayers(background=roi_img_volume, fit=True)
        
    def onVolumeSelected(self):
        print('Volume selected')
        self.currentVolume = self.ui.volumeComboBox.currentNode()


    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return
        
        self.ui.volumeComboBox.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume"))


        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return
    

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        nodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
        

        self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.volumeComboBox.currentNodeID)


 
        self._parameterNode.EndModify(wasModified)


class LungNoduleROILogic(ScriptedLoadableModuleLogic):

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        if not parameterNode.GetParameter("Threshold"):
            parameterNode.SetParameter("Threshold", "100.0")
        if not parameterNode.GetParameter("Invert"):
            parameterNode.SetParameter("Invert", "false")

    def process(self, inputVolume, outputVolume, imageThreshold, invert=False, showResult=True):
        pass

    def calculate_nodule_ROI(self):
        pass

class LungNoduleROITest(ScriptedLoadableModuleTest):

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_t_ApplyThreshold1()

    def test_t_ApplyThreshold1(self):

        pass

class ImageROI:
    def __init__(self):
        print("ImageROI object created")

    # convert coordinates from slicer spacing to sitk spacing
    # inputs:
    # img -> SimpleITK image
    # slicer_size -> size from slicer resampling, found in volume metadata in slicer
    # slicer_coords -> (x,y,z) coordinates of point in slicer
    def convert_slicer_coordinates(self, img, slicer_size, slicer_coords):
        size_conversion = np.divide(img.GetSize(), slicer_size)
        python_coords_dbl = np.multiply(slicer_coords, size_conversion)
        
        python_coords = []
        for coord in python_coords_dbl:
            python_coords.append(int(coord))
        
        return python_coords

    # create roi image from sitk image
    # inputs:
    # img -> SimpleITK image
    # centroid -> centroid of lung nodule in coordinates using [1,1,1] spacing NOT SLICER
    # expansion -> amount to expand ROI from centroid in +/- for each direction
    def create_roi_image(self, img, expansion, centroid):
        # expand roi from centroid
        roi = {
            'coronal' : [centroid[0]-expansion, centroid[0]+expansion],
            'sagittal' : [centroid[1]-expansion, centroid[1]+expansion],
            'axial' : [centroid[2]-expansion, centroid[2]+expansion]
        }

        # convert to numpy array and cut down to roi around nodule centroid
        np_img = sitk.GetArrayFromImage(img)
        np_roi = np_img[roi['axial'][0]:roi['axial'][1],
                        roi['sagittal'][0]:roi['sagittal'][1],
                        roi['coronal'][0]:roi['coronal'][1]]

        # return a SimpleITK image bounded in ROI
        print('ROI shape:', np_roi.shape)
        return np_roi

    # basic function for resampling an image assuming dcm is already identity matrix
    # inputs:
    # img -> SimpleITK image to be resampled
    def resample_image(self, img, spacing):
        resample = sitk.ResampleImageFilter()

        k = np.divide([1,1,1], img.GetSpacing())
        
        resample_size_float = np.divide(img.GetSize(), k)
        resample_spacing_float = np.multiply(img.GetSpacing(), k)
        
        resample_size = []
        resample_spacing = spacing
        
        for idx, dim in enumerate(resample_size_float):
            resample_size.append(int(dim))
        
        resample = sitk.ResampleImageFilter()
        resample.SetOutputOrigin(img.GetOrigin())
        resample.SetSize(resample_size)
        resample.SetOutputSpacing(resample_spacing)
        resample.SetOutputDirection(img.GetDirection())
        resample.SetInterpolator(sitk.sitkLinear)
        
        return resample.Execute(img)
