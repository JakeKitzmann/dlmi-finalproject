import logging
import os

import vtk

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

import numpy as np
import SimpleITK as sitk
import sitkUtils

import glob
import csv



class LungNoduleROI(ScriptedLoadableModule):

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Lung Nodule ROI"  
        self.parent.categories = ["Deep Learning Lung Nodule Segmentation"] 
        self.parent.dependencies = [] 
        self.parent.contributors = ["Jake Kitzmann (Advanced Pulmonary Physiomic Imaging Laboratory -- University of Iowa Roy J. and Lucille H. Carver College of Medicine)"]

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
        self.inSlices = False

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
        self.ui.batchCaseApplyButton.connect('clicked(bool)', self.onBatchCaseApplyButton)

        # Sliders
        self.ui.roiSizeSlider.minimum = 4
        self.ui.roiSizeSlider.maximum = 35
        self.ui.roiSizeSlider.value = 10
        self.ui.roiSizeSlider.connect('valueChanged(int)', self.onRoiSliderValueChanged)

        self.ui.roiSizeLabel.connect('textChanged(QString)', self.userChangedRoiSize)
        self.ui.roiSizeLabel.setText(f'{self.ui.roiSizeSlider.value}')
        self.ui.centroidManualButton.connect('clicked(bool)', self.onCentroidManualButton)

        # non isotropic size sliders

        nonIsoSliders = [self.ui.sSliderNonIso, self.ui.cSliderNonIso, self.ui.aSliderNonIso]

        nonIsoSliders[0].connect('valueChanged(int)', self.sSliderNonIsoChanged)
        nonIsoSliders[1].connect('valueChanged(int)', self.cSliderNonIsoChanged)
        nonIsoSliders[2].connect('valueChanged(int)', self.aSliderNonIsoChanged)

        for slider in nonIsoSliders:
            slider.minimum = 4
            slider.maximum = 35
            slider.value = 10

        # LineEdits non iso
        self.ui.sLineEditNonIso.text = self.ui.sSliderNonIso.value
        self.ui.cLineEditNonIso.text = self.ui.cSliderNonIso.value
        self.ui.aLineEditNonIso.text = self.ui.aSliderNonIso.value

        self.ui.aLineEditNonIso.connect('textChanged(QString)', self.aLineEditNonIsoChanged)
        self.ui.cLineEditNonIso.connect('textChanged(QString)', self.cLineEditNonIsoChanged)
        self.ui.sLineEditNonIso.connect('textChanged(QString)', self.sLineEditNonIsoChanged)

        self.ui.roiSizeLabel.setText(f'{self.ui.roiSizeSlider.value * 2}')

        # Combobox
        self.ui.volumeComboBox.setMRMLScene(slicer.mrmlScene)
        self.ui.volumeComboBox.connect('currentNodeChanged(vtkMRMLNode*)', self.onVolumeSelected)
        self.ui.volumeComboBox.renameEnabled = True

        # CheckBoxes
        self.ui.roiCheckBox.connect('clicked(bool)', self.onRoiCheckBox)
        self.onRoiCheckBox()

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

        # Radio Buttons
        self.ui.singleCaseRadioButton.setChecked(True)
        self.onSingleCaseRadioButton()        
        self.ui.singleCaseRadioButton.connect('clicked(bool)', self.onSingleCaseRadioButton)
        self.ui.batchCaseRadioButton.connect('clicked(bool)', self.onBatchCaseRadioButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()


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

    def onSingleCaseRadioButton(self):
        self.ui.batchCase.setVisible(False)
        self.ui.singleCase.setVisible(True)

    def onBatchCaseRadioButton(self):
        self.ui.singleCase.setVisible(False)
        self.ui.batchCase.setVisible(True)

    def aSliderNonIsoChanged(self):
        self.ui.aLineEditNonIso.text = self.ui.aSliderNonIso.value * 2

    def cSliderNonIsoChanged(self):
        self.ui.cLineEditNonIso.text = self.ui.cSliderNonIso.value * 2

    def sSliderNonIsoChanged(self):
        self.ui.sLineEditNonIso.text = self.ui.sSliderNonIso.value * 2

    def aLineEditNonIsoChanged(self):
        self.ui.aSliderNonIso.value = (int(self.ui.aLineEditNonIso.text)) / 2

    def cLineEditNonIsoChanged(self):
        self.ui.cSliderNonIso.value = (int(self.ui.cLineEditNonIso.text)) / 2

    def sLineEditNonIsoChanged(self):
        self.ui.sSliderNonIso.value = (int(self.ui.sLineEditNonIso.text)) / 2
    
    def onRoiCheckBox(self):
        gridLayoutComponents = [self.ui.sLabelNonIso, self.ui.sLineEditNonIso, self.ui.sSliderNonIso, self.ui.cLabelNonIso, self.ui.cLineEditNonIso, self.ui.cSliderNonIso, self.ui.aLabelNonIso, self.ui.aLineEditNonIso, self.ui.aSliderNonIso]
        
        if self.ui.roiCheckBox.isChecked():
           for component in gridLayoutComponents:
                component.setVisible(True)
        else:
            for component in gridLayoutComponents:
                component.setVisible(False)


    def clearNoduleCentroids(self):
        nodeList = slicer.util.getNodesByClass('vtkMRMLMarkupsFiducialNode')
        if len(nodeList) != 0:
            for node in nodeList:
                if node.GetName() == 'nodule_centroid':
                    slicer.mrmlScene.RemoveNode(node)

    def onNoduleCentroidButton(self):
        self.clearNoduleCentroids()

    def onNoduleCentroidButton(self):
        fiducialNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode')
        fiducialNode.SetName('nodule_centroid')

        slicer.modules.markups.logic().StartPlaceMode(0)        
        self.inSlices = False

    def onCentroidManualButton(self):
        self.clearNoduleCentroids()

        fiducialNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode')
        fiducialNode.SetName('nodule_centroid')
        img = self.ui.volumeComboBox.currentNode()
        slices = [int(self.ui.sLineEdit.text), int(self.ui.cLineEdit.text), int(self.ui.aLineEdit.text)]
        print(f'slices: {slices}')

        self.inSlices = True


    def onRoiSliderValueChanged(self):
        self.ui.roiSizeLabel.setText(f'{self.ui.roiSizeSlider.value * 2 }')

    def userChangedRoiSize(self):
        self.ui.roiSizeSlider.value = (int(self.ui.roiSizeLabel.text) / 2)


    def onRoiSliderValueChanged(self):
        self.ui.roiSizeLabel.setText(f'{self.ui.roiSizeSlider.value * 2}')


    def onApplyButton(self):

        volume = self.ui.volumeComboBox.currentNode()
        node = slicer.mrmlScene.GetFirstNodeByName('nodule_centroid')

        if volume is None:
            logging.error('No volume selected')
            return  

        if node is None:
            logging.error('No centroid selected')
            return
    
        if not self.inSlices:
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

            self.ui.sLineEdit.text = str(difference_slices_int[0])
            self.ui.cLineEdit.text = str(difference_slices_int[1])
            self.ui.aLineEdit.text = str(difference_slices_int[2])

        else:
            difference_slices_int = [int(self.ui.sLineEdit.text), int(self.ui.cLineEdit.text), int(self.ui.aLineEdit.text)]
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

        size = [0,0,0]

        if self.ui.roiCheckBox.isChecked():
            size[0] = int(self.ui.sLineEditNonIso.text)
            size[1] = int(self.ui.cLineEditNonIso.text)
            size[2] = int(self.ui.aLineEditNonIso.text)
            print(f'Non-isotropic size: {size}')
        else:
            size = [self.ui.roiSizeSlider.value * 2, self.ui.roiSizeSlider.value * 2, self.ui.roiSizeSlider.value * 2 ]
            print(f'Isotropic size: {size}')

        # create roi
        self.create_roi(volume, difference_slices_int, size)

    def create_roi(self, volume, centroid, size):

        sitk_img = sitkUtils.PullVolumeFromSlicer(volume.GetID())
        imageROI = ImageROI()

        print(f'Creating ROI with size {size} and centroid {centroid}')


        roi_img_np = imageROI.create_roi_image(sitk_img, size, centroid)

        roi_img_volume = slicer.util.addVolumeFromArray(roi_img_np)
        roi_img_volume.SetName(self.ui.fileName.text)

        if not self.ui.interpolationCheckBox.isChecked():
            roi_img_display_node = roi_img_volume.GetDisplayNode()
            roi_img_display_node.SetInterpolate(0)

        slicer.util.setSliceViewerLayers(background=roi_img_volume, fit=True)
        print('ROI created')

        return roi_img_volume

    def onVolumeSelected(self):
        self.currentVolume = self.ui.volumeComboBox.currentNode()

    # BATCH CASES

    def onBatchCaseApplyButton(self):
        volumeListPath = self.ui.batchVolumeLineEdit.text
        centroidListPath = self.ui.batchCentroidLineEdit.text
        outputDir = self.ui.batchOutputLineEdit.text


        volumePaths = os.listdir(volumeListPath)

        cases = []

        with open(centroidListPath) as file_obj: 
            reader_obj = csv.reader(file_obj)
            rows = []
            for row in reader_obj:
                print(row)
                rows.append(row)

        # create pairs of volumes and centroids for each case
        for volumePath in volumePaths:
            print('-----------------' + volumePath)
            pid = volumePath.split('/')[-1].split('_')[0]

            for row in rows:
                if pid == row[0]:
                    print('match')
                    cases.append(self.case(volumeListPath + volumePath, row[0], row[1], row[2], row[3], row[4]))



        for case in cases:
            volume = slicer.util.loadVolume(case.volumePath)
            centroid = [int(case.centroidS), int(case.centroidC), int(case.centroidA)]
            size = [int(case.size), int(case.size), int(case.size)]

            roi = self.create_roi(volume, centroid, size)

            success = slicer.util.saveNode(roi, outputDir + '/' + case.PID + '_roi.nrrd')
            if success:
                print(f'Volume {case.PID} saved to {outputDir}')
            else:
                print(f'Failed to save volume {case.PID} to {outputDir}')
               
    

    # class to store case information in batch processing
    class case:
        def __init__(self,volumePath, PID, centroidS, centroidC, centroidA, size):
            self.volumePath = volumePath
            self.PID = PID
            self.centroidS = centroidS
            self.centroidC = centroidC
            self.centroidA = centroidA
            self.size = size


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
    # create roi image from sitk image
    # inputs:
    # img -> SimpleITK image
    # centroid -> centroid of lung nodule in coordinates using [1,1,1] spacing NOT SLICER
    # expansion -> amount to expand ROI from centroid in +/- for each direction
    def create_roi_image(self, img, expansion, centroid):
        # expand roi from centroid
        roi = {
            'coronal' : [centroid[1]-int((expansion[1]) / 2), centroid[1]+int((expansion[1]) / 2)],
            'sagittal' : [centroid[0]-int((expansion[0]) / 2), centroid[0]+int((expansion[0]) / 2)],
            'axial' : [centroid[2]-int((expansion[2])/ 2), centroid[2]+int((expansion[2]) / 2)]

        }

        # convert to numpy array and cut down to roi around nodule centroid
        np_img = sitk.GetArrayFromImage(img)
        np_roi = np_img[roi['axial'][0]:roi['axial'][1],
                        roi['coronal'][0]:roi['coronal'][1],
                        roi['sagittal'][0]:roi['sagittal'][1]]

        return np_roi