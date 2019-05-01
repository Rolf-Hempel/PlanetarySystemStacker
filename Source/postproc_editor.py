# -*- coding: utf-8; -*-
"""
Copyright (c) 2018 Rolf Hempel, rolf6419@gmx.de

This file is part of the PlanetarySystemStacker tool (PSS).
https://github.com/Rolf-Hempel/PlanetarySystemStacker

PSS is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PSS.  If not, see <http://www.gnu.org/licenses/>.

"""

from copy import deepcopy
from os.path import splitext
from sys import argv, stdout
from time import sleep

from PyQt5 import QtWidgets, QtCore
from cv2 import imread, cvtColor, COLOR_BGR2GRAY
from math import sqrt

from frame_viewer import FrameViewer
from frames import Frames
from miscellaneous import Miscellaneous
from postproc_editor_gui import Ui_postproc_editor
from sharpening_layer_widget import Ui_sharpening_layer_widget
from version_manager_widget import Ui_version_manager_widget


class DataObject(object):
    """
    This class implements the central data object used in postprocessing.

    """

    def __init__(self, image_original, name_original, suffix, blinking_period, idle_loop_time):
        """
        Initialize the data object.

        :param image_original: Image file (16bit Tiff) holding the input for postprocessing
        :param name_original: Path name of the original image.
        :param suffix: File suffix (str) to be appended to the original path name. The resulting
                       path is used to store the image resulting from postprocessing.
        :param blinking_period: Time between image exchanges in the blink comparator
        :param idle_loop_time: The image processor is realized as a separate thread. It checks
                               periodically if a new image version is to be computed. This parameter
                               specifies the idle time between checks.
        """

        self.image_original = image_original
        self.color = len(self.image_original.shape) == 3
        self.file_name_original = name_original
        self.file_name_processed = splitext(name_original)[0] + suffix + '.tiff'

        # Initialize the postprocessing image versions with the unprocessed image (as version 0).
        self.versions = [Version(self.image_original)]
        self.number_versions = 0

        # Initialize the pointer to the currently selected version to 0 (input image).
        # "version_compared" is used by the blink comparator later on. The blink comparator is
        # switched off initially.
        self.version_selected = 0
        self.version_compared = 0
        self.blinking = False
        self.blinking_period = blinking_period
        self.idle_loop_time = idle_loop_time

        # Create a first processed version with initial parameters for Gaussian radius. The amount
        # of sharpening is initialized to zero.
        initial_version = Version(self.image_original)
        initial_version.add_layer(Layer(1., 0, False))
        self.add_version(initial_version)

    def add_version(self, version):
        """
        Add a new postprocessing version, and set the "selected" pointer to it.

        :param version:
        :return: -
        """

        self.versions.append(version)
        self.number_versions += 1
        self.version_selected = self.number_versions

    def remove_version(self, index):
        """
        Remove a postprocessing version, and set the "selected" pointer to the previous one. The
        initial version (input image) cannot be removed.

        :param index: Index of the version to be removed.
        :return: -
        """

        if 0 < index <= self.number_versions:
            self.versions = self.versions[:index] + self.versions[index + 1:]
            self.number_versions -= 1
            self.version_selected = index - 1


class Version(object):
    """
    Instances of this class hold the data defining a single postprocessing version, including the
    resulting image for the current parameter set.
    """

    def __init__(self, image):
        """
        Initialize the version object with the input image and an empty set of processing layers.
        :param image: Input image (16bit Tiff) for postprocessing
        """

        self.image = image
        self.layers = []
        self.number_layers = 0

    def add_layer(self, layer):
        """
        Add a postprocessing layer.
        :param layer: Layer instance to be added to the list of layers.
        :return: -
        """

        self.layers.append(layer)
        self.number_layers += 1

    def remove_layer(self, layer_index):
        """
        Remove a postprocessing layer from this version.

        :param layer_index: Index of the layer to be removed.
        :return: -
        """

        if 0 <= layer_index < self.number_layers:
            self.layers = self.layers[:layer_index] + self.layers[layer_index + 1:]
            self.number_layers -= 1


class Layer(object):
    """
    Instances of this class hold the parameters which define a postprocessing layer.
    """

    def __init__(self, radius, amount, luminance_only):
        """
        Initialize the Layer instance with values for Gaussian radius, amount of sharpening and a
        flag which indicates on which channel the sharpening is to be applied.

        :param radius: Radius (in pixels) of the Gaussian sharpening kernel.
        :param amount: Amount of sharpening for this layer.
        :param luminance_only: True, if sharpening is to be applied to the luminance channel only.
                               False, otherwise.
        """

        self.radius = radius
        self.amount = amount
        self.luminance_only = luminance_only


class SharpeningLayerWidget(QtWidgets.QWidget, Ui_sharpening_layer_widget):
    """
    GUI widget to manipulate the parameters of a sharpening layer. Four instances are created, so
    up to four sharpening layers can be defined.
    """

    def __init__(self, layer_index, remove_layer_callback, parent=None):
        """
        Initialize the widget.

        :param layer_index: Index of the sharpening layer (starting with 1).
        :param remove_layer_callback: Callback function to remove a layer from the current version.
        :param parent: Parent object.
        """

        QtWidgets.QWidget.__init__(self, parent)
        self.setupUi(self)

        self.layer_index = layer_index
        self.title = "Layer " + str(layer_index + 1)
        self.remove_layer_callback = remove_layer_callback

        self.horizontalSlider_radius.valueChanged.connect(self.horizontalSlider_radius_changed)
        self.lineEdit_radius.textChanged.connect(self.lineEdit_radius_changed)
        self.horizontalSlider_amount.valueChanged.connect(self.horizontalSlider_amount_changed)
        self.lineEdit_amount.textChanged.connect(self.lineEdit_amount_changed)
        self.checkBox_luminance.stateChanged.connect(self.checkBox_luminance_toggled)
        self.pushButton_remove.clicked.connect(self.remove_layer)

    def set_values(self, layer):
        """
        Set the values of the GUI widget according to the current data model

        :param layer: Layer instance holding the sharpening parameters
        :return: -
        """

        self.layer = layer

        self.groupBox_layer.setTitle(self.title)
        self.horizontalSlider_radius.setValue(self.radius_to_integer(self.layer.radius))
        self.lineEdit_radius.setText(str(self.layer.radius))
        self.horizontalSlider_amount.setValue(self.amount_to_integer(self.layer.amount))
        self.lineEdit_amount.setText(str(self.layer.amount))
        self.checkBox_luminance.setChecked(self.layer.luminance_only)

    # The following four methods implement the translation between data model values and the
    # (integer) values of the horizontal GUI sliders. In the case of the sharpening amount this
    # translation is non-linear in order to add resolution at small values.
    @staticmethod
    def radius_to_integer(radius):
        return max(min(int(round(radius * 10.)), 99), 1)

    @staticmethod
    def integer_to_radius(int):
        return int / 10.

    @staticmethod
    def amount_to_integer(amount):
        return max(0, min(int(round(sqrt(50. * amount))), 100))

    @staticmethod
    def integer_to_amount(integer):
        return 0.02 * integer ** 2

    def horizontalSlider_radius_changed(self):
        """
        The "radius" slider has been moved. Update the corresponding lineEdit widget. Block its
        signals temporarily to avoid cross-talk.
        :return: -
        """

        self.layer.radius = self.integer_to_radius(self.horizontalSlider_radius.value())
        self.lineEdit_radius.blockSignals(True)
        self.lineEdit_radius.setText(str(self.layer.radius))
        self.lineEdit_radius.blockSignals(False)

    def lineEdit_radius_changed(self):
        """
        The text of the lineEdit widget has been changed. Update the horizontal slider. Include a
        try-except block to avoid illegal user input to crash the program.

        :return: -
        """

        try:
            self.layer.radius = max(0.1, min(float(self.lineEdit_radius.text()), 9.9))
            self.horizontalSlider_radius.blockSignals(True)
            self.horizontalSlider_radius.setValue(self.radius_to_integer(self.layer.radius))
            self.horizontalSlider_radius.blockSignals(False)
        except:
            pass

    def horizontalSlider_amount_changed(self):
        """
        The same as above for the "amount" parameter.

        :return: -
        """

        self.layer.amount = self.integer_to_amount(self.horizontalSlider_amount.value())
        self.lineEdit_amount.blockSignals(True)
        self.lineEdit_amount.setText("{0:.2f}".format(round(self.layer.amount, 2)))
        self.lineEdit_amount.blockSignals(False)

    def lineEdit_amount_changed(self):
        """
        The same as above for the "amount" parameter.

        :return: -
        """

        try:
            self.layer.amount = float(self.lineEdit_amount.text())
            self.horizontalSlider_amount.blockSignals(True)
            self.horizontalSlider_amount.setValue(self.amount_to_integer(self.layer.amount))
            self.horizontalSlider_amount.blockSignals(False)
        except:
            pass

    def checkBox_luminance_toggled(self):
        """
        The checkbox which states if the sharpening should affect the luminance channel only has
        changed its state.

        :return: -
        """

        self.layer.luminance_only = not self.layer.luminance_only

    def remove_layer(self):
        """
        This method is executed when the user presses the "remove" button. The actual action is
        performed by the higher-level "PostprocEditorWidget" object.
        :return: -
        """

        self.remove_layer_callback(self.layer_index)


class VersionManagerWidget(QtWidgets.QWidget, Ui_version_manager_widget):
    """
    This GUI widget manages the different postprocessing versions and controls which image is
    displayed in the image viewer. It also starts and controls the blink comparator.
    """

    # The set_photo_signal tells the image viewer which image it should show.
    set_photo_signal = QtCore.pyqtSignal(int)
    # The blink comparator emits the variant_shown_signal to highlight / de-highlight the GUI image
    # selector which corresponds to the image currently displayed.
    variant_shown_signal = QtCore.pyqtSignal(bool, bool)

    def __init__(self, data_object, select_version_callback, parent=None):
        """

        :param data_object: Central data object holding postprocessing images and parameters.
        :param select_version_callback: Higher-level method to set the currently selected version.
        :param parent: parent object.
        """

        QtWidgets.QWidget.__init__(self, parent)
        self.setupUi(self)
        self.data_object = data_object
        self.select_version_callback = select_version_callback

        self.spinBox_version.valueChanged.connect(self.select_version)
        self.spinBox_compare.valueChanged.connect(self.select_version_compared)
        self.pushButton_new.clicked.connect(self.new_version)
        self.pushButton_delete.clicked.connect(self.remove_version)
        self.checkBox_blink_compare.stateChanged.connect(self.blinking_toggled)
        self.pushButton_save.clicked.connect(self.save_version)
        self.pushButton_save_as.clicked.connect(self.save_version_as)
        self.variant_shown_signal.connect(self.highlight_variant)

        # Set the ranges for version selection spinboxes. Initially, there are only two versions.
        self.spinBox_version.setMaximum(1)
        self.spinBox_version.setMinimum(0)
        self.spinBox_compare.setMaximum(1)
        self.spinBox_compare.setMinimum(0)

        # Set the spinbox to the newly created version
        self.spinBox_version.setValue(self.data_object.version_selected)

    def select_version(self):
        """
        This method is called when the user changes the setting of the "version selected" spinbox.

        :return: -
        """
        self.data_object.version_selected = self.spinBox_version.value()
        self.select_version_callback(self.data_object.version_selected)

    def select_version_compared(self):
        """
        This method is called when the user changes the setting of the "version compared" spinbox.
        The "selected" and "compared" versions are displayed alternately by the blink comparator.
        :return: -
        """

        self.data_object.version_compared = self.spinBox_compare.value()

    def new_version(self):
        """
        Create a new postprocessing version.
        :return: -
        """

        # Create a Version object, and add an initial layer (radius 1., amount 0).
        new_version = Version(self.data_object.image_original)
        new_version.add_layer(Layer(1., 0, False))
        self.data_object.add_version(new_version)

        # Set the image viewer to the new version, and increase the range of spinboxes to include
        # the new version.
        self.set_photo_signal.emit(self.data_object.version_selected)
        self.spinBox_version.setMaximum(self.data_object.number_versions)
        self.spinBox_version.setValue(self.data_object.number_versions)
        self.spinBox_compare.setMaximum(self.data_object.number_versions)

    def remove_version(self):
        """
        Remove a postprocessing version.
        :return: -
        """

        # Remove the version from the central data object, and instruct the image viewer to
        # show the previous version.
        self.data_object.remove_version(self.data_object.version_selected)
        self.set_photo_signal.emit(self.data_object.version_selected)

        # Adjust the spinbox bounds, and set the current version parameters to the new selection.
        self.spinBox_version.setMaximum(self.data_object.number_versions)
        self.spinBox_compare.setMaximum(self.data_object.number_versions)
        self.select_version_callback(self.data_object.version_selected)

    def blinking_toggled(self):
        """
        Switch the blink comparator on or off.

        :return: -
        """

        # Toggle the status variable.
        self.data_object.blinking = not self.data_object.blinking

        # The blink comparator is switched on.
        if self.data_object.blinking:
            # Create the blink comparator thread and start it.
            self.blink_comparator = BlinkComparator(self.data_object, self.set_photo_signal,
                                                    self.variant_shown_signal)
            self.blink_comparator.setTerminationEnabled(True)

        # The blink comparator is switched off. Set the image viewer to the currently selected
        # image and terminate the separate thread.
        else:
            self.set_photo_signal.emit(self.data_object.version_selected)
            self.blink_comparator.stop()

    def highlight_variant(self, selected, compare):
        """
        Whenever the blink comparator changes the display, the spinbox corressponding to the
        currently displayed image changes its font color to red. The other spinbox is blanked out
        (by setting font color to white). When the blink comparator is not active, both spinboxes
        show their numbers in bladk font.

        :param selected: True, if the image corresponding to the "selected" spinbox is displayed.
                         False, otherwise.
        :param compare: True, if the image corresponding to the "compared" spinbox is displayed.
                        False, otherwise.
        :return: -
        """

        if selected and compare:
            self.spinBox_version.setStyleSheet('color: red')
            self.spinBox_compare.setStyleSheet('color: red')
        elif not selected and not compare:
            self.spinBox_version.setStyleSheet('color: black')
            self.spinBox_compare.setStyleSheet('color: black')
        elif selected:
            self.spinBox_version.setStyleSheet('color: red')
            self.spinBox_compare.setStyleSheet('color: white')
        elif compare:
            self.spinBox_version.setStyleSheet('color: white')
            self.spinBox_compare.setStyleSheet('color: red')

    def save_version(self):
        """
        save the result as 16bit Tiff at the standard location.

        :return: -
        """

        Frames.save_image(self.data_object.file_name_processed,
                          self.data_object.versions[self.data_object.version_selected].image,
                          color=self.data_object.color, avoid_overwriting=False)

    def save_version_as(self):
        """
        save the result as 16bit Tiff at a location selected by the user.

        :return: -
        """

        options = QtWidgets.QFileDialog.Options()
        filename, extension = QtWidgets.QFileDialog.getSaveFileName(self,
                            "Save result as 16bit Tiff image", self.data_object.file_name_original,
                            "Image Files (*.tiff)", options=options)

        if filename and extension:
            Frames.save_image(filename,
                              self.data_object.versions[self.data_object.version_selected].image,
                              color=self.data_object.color, avoid_overwriting=False)


class BlinkComparator(QtCore.QThread):
    """
    This class implements the blink comparator. It shows alternately two image versions.
    """

    def __init__(self, data_object, set_photo_signal, variant_shown_signal, parent=None):
        """
        :param data_object: The central postprocessing data object.
        :param set_photo_signal: Signal which tells the image viewer which image it should show.
        :param variant_shown_signal: Signal which tells the GUI widget which spinbox should be
                                     highlighted.
        :param parent: Parent object
        """

        QtCore.QThread.__init__(self, parent)
        self.data_object = data_object
        self.set_photo_signal = set_photo_signal
        self.variant_shown_signal = variant_shown_signal

        # Initially, no version is shown by the blink comparator.
        self.variant_shown_signal.emit(False, False)

        # Start the comparator thread.
        self.start()

    def run(self):
        """
        Blink comparator main loop. As long as the "blinking" status variable is True, show the
        two image versions alternately.

        :return: -
        """

        # Begin with showing the "selected" version.
        show_selected_version = True

        # Begin main loop.
        while self.data_object.blinking:
            # Show the "selected" version.
            if show_selected_version:
                self.set_photo_signal.emit(self.data_object.version_selected)
                self.variant_shown_signal.emit(True, False)
            # Show the "compared" version.
            else:
                self.set_photo_signal.emit(self.data_object.version_compared)
                self.variant_shown_signal.emit(False, True)

            # Toggle back and forth between first and second image version.
            show_selected_version = not show_selected_version

            # Sleep time inserted to limit CPU consumption by idle looping.
            sleep(self.data_object.blinking_period)

    def stop(self):
        """
        Terminate the blink comparator thread.

        :return: -
        """

        self.variant_shown_signal.emit(False, False)
        self.terminate()


class ImageProcessor(QtCore.QThread):
    """
    This class implements the asynchroneous computation of new image versions.
    """

    # The set_photo_signal tells the image viewer which image it should show.
    set_photo_signal = QtCore.pyqtSignal(int)
    # The set_status_bar_signal triggers the display of a (colored) message in the main GUI's
    # status bar.
    set_status_bar_signal = QtCore.pyqtSignal(str, str)

    def __init__(self, data_object, parent=None):
        """
        Whenever the parameters of the current version change, compute a new sharpened image.

        :param data_object: Data object with postprocessing data
        """

        QtCore.QThread.__init__(self, parent)
        self.data_object = data_object

        # Remember the last version (and the corresponding layer parameters) for which an image
        # was computed. As soon as this data change, a new image is computed.
        self.last_version_selected = 1
        self.last_layers = [Layer(1., 0, False)]

        # Initialize the data objects holding the currently active version.
        self.version_selected = None
        self.layers_selected = None

        self.start()

    def run(self):
        while True:
            # To avoid chasing a moving target, copy the parameters of the currently active version
            self.version_selected = self.data_object.version_selected
            self.layers_selected = deepcopy(self.data_object.versions[
                                                self.data_object.version_selected].layers)

            # Compare the currently active version with the last one for which an image was
            # computed. If there is a difference, start a new computation.
            if self.new_computation_required() and self.version_selected:

                # Change the main GUI's status bar to show that a computation is going on.
                self.set_status_bar_signal.emit(
                    "Processing " + self.data_object.file_name_original +
                    ", busy computing a new postprocessing image.", "black")

                # Perform the new computation.
                self.recompute_selected_version()

                # Reset the status bar to its idle state.
                self.set_status_bar_signal.emit(
                    "Processing " + self.data_object.file_name_original + ", postprocessing.",
                    "black")

                # Show the new image in the image viewer, and remember its parameters.
                self.set_photo_signal.emit(self.version_selected)
                self.last_version_selected = self.version_selected
                self.last_layers = self.layers_selected

            # Idle loop before doing the next check for updates.
            else:
                sleep(self.data_object.idle_loop_time)

    def new_computation_required(self):
        """
        Decide if a new computation is required.

        :return: True, if the current version parameters have changed. False, otherwise.
        """

        # First check if another version was selected, or if the number of layers has changed.
        if self.last_version_selected != self.version_selected or \
                len(self.last_layers) != len(self.layers_selected):
            return True

        # For all layers check if a parameter has changed.
        for last_layer, layer_selected in zip(self.last_layers, self.layers_selected):
            if last_layer.radius != layer_selected.radius or \
                    last_layer.amount != layer_selected.amount or \
                    last_layer.luminance_only != layer_selected.luminance_only:
                return True

        # No change detected.
        return False

    def recompute_selected_version(self):
        """
        Do the actual computation. Starting from the original image, for each sharpening layer
        apply a Gaussian filter using the layer's parameters. Store the result in the central
        data object for the selected version.

        :return: -
        """

        # Initialize the new image with the original image.
        new_image = self.data_object.image_original

        # Apply all sharpening layers.
        for layer in self.layers_selected:
            new_image = Miscellaneous.gaussian_sharpen(new_image, layer.amount, layer.radius,
                                                       luminance_only=layer.luminance_only)

        # Store the result in the central data object.
        self.data_object.versions[self.data_object.version_selected].image = new_image

    def stop(self):
        """
        Terminate the ImageProcessor.
        :return: -
        """

        self.terminate()


class PostprocEditorWidget(QtWidgets.QFrame, Ui_postproc_editor):
    """
    This widget implements a frame viewer together with control elements to control the
    postprocessing. Several postprocessing versions can be created and managed, each one using
    up to four sharpening layers. A blink comparator allows comparing any two versions.
    """

    def __init__(self, image_original, name_original, suffix, blinking_period, idle_loop_time,
                 set_status_bar_callback):
        """

        :param image_original: Image file (16bit Tiff) holding the input for postprocessing
        :param name_original: Path name of the original image.
        :param suffix: File suffix (str) to be appended to the original path name. The resulting
                       path is used to store the image resulting from postprocessing.
        :param blinking_period: Time between image exchanges in the blink comparator
        :param idle_loop_time: The image processor is realized as a separate thread. It checks
                               periodically if a new image version is to be computed. This parameter
                               specifies the idle time between checks.
        :param set_status_bar_callback: Call-back function to update the main GUI's status bar.
        """

        super(PostprocEditorWidget, self).__init__()
        self.setupUi(self)

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.pushButton_add_layer.clicked.connect(self.add_layer)

        self.label_message.setText("Create sharp image versions using up to four correction layers."
                                   " Adjust layer parameters as required.")
        self.label_message.setStyleSheet('color: red')

        # Start the frame viewer.
        self.frame_viewer = FrameViewer()
        self.frame_viewer.setObjectName("framewiever")
        self.gridLayout.addWidget(self.frame_viewer, 0, 0, 7, 1)

        # Create and initialize the central data object.
        self.data_object = DataObject(image_original, name_original, suffix, blinking_period,
                                      idle_loop_time)

        # Create four widgets to control the parameters of individual sharpening layers. Widgets
        # corresponding to inactive layers will be de-activated.
        self.sharpening_layer_widgets = []
        self.max_layers = 4
        for layer in range(self.max_layers):
            sharpening_layer_widget = SharpeningLayerWidget(layer, self.remove_layer)
            self.gridLayout.addWidget(self.frame_viewer, 0, 0, 7, 1)
            self.gridLayout.addWidget(sharpening_layer_widget, layer + 1, 1, 1, 1)
            self.sharpening_layer_widgets.append(sharpening_layer_widget)

        # Create the version manager and pass it the "select_version" callback function.
        self.version_manager_widget = VersionManagerWidget(self.data_object, self.select_version)
        self.gridLayout.addWidget(self.version_manager_widget, 6, 1, 1, 1)

        # The "set_photo_signal" from the VersionManagerWidget is not passed to the image viewer
        # directly. (The image viewer does not accept signals.) Instead, it calls the "select_image"
        # method below which in turn invokes the image viewer.
        self.version_manager_widget.set_photo_signal.connect(self.select_image)

        # Select the initial current version.
        self.select_version(self.data_object.version_selected)

        # Start the image processor thread, and connect its signals.
        self.image_processor = ImageProcessor(self.data_object)
        self.image_processor.setTerminationEnabled(True)
        self.image_processor.set_photo_signal.connect(self.select_image)
        self.image_processor.set_status_bar_signal.connect(set_status_bar_callback)

    def select_version(self, version_index):
        """
        Select a new current version and update all layer widgets.

        :param version_index: Index of the version selected in version list of the central data
                              object.
        :return: -
        """

        version_selected = self.data_object.versions[version_index]

        # Set the layer widgets values for all active layers. Hide the inactive layer widgets.
        for layer_index, layer in enumerate(version_selected.layers):
            self.sharpening_layer_widgets[layer_index].set_values(layer)
            self.sharpening_layer_widgets[layer_index].setHidden(False)
        for layer_index in range(version_selected.number_layers, self.max_layers):
            self.sharpening_layer_widgets[layer_index].setHidden(True)

        # Load the current image into the image viewer.
        self.select_image(version_index)

    def select_image(self, version_index):
        """
        Load the current image into the image viewer. This method is either invoked by a direct
        call or by a "set_photo_signal" signal.

        :param version_index: Index of the version selected in version list of the central data
                              object.
        :return: -
        """

        self.frame_viewer.setPhoto(self.data_object.versions[version_index].image)

    def add_layer(self):
        """
        This method is called when the "Add postprocessing layer" button is pressed. Add a layer
        (up to a maximum of four) to the currently selected postprocessing version.

        :return:-
        """

        version_selected = self.data_object.versions[self.data_object.version_selected]
        num_layers_current = version_selected.number_layers

        # The "zero" layer is reserved for the original image, so for this version no additional
        # layer can be allocated.

        # The first layer is initialized with a fixed parameter set (see "init" method of DataObject
        # class).
        if self.data_object.version_selected and num_layers_current < self.max_layers:

            # This version has already at least one layer. Initialize the radius parameter of the
            # new layer to 1.5 times the radius of the previous one.
            if num_layers_current > 0:
                previous_layer = version_selected.layers[num_layers_current - 1]
                new_layer = Layer(1.5 * previous_layer.radius, 0, previous_layer.luminance_only)

            # This is the first layer for this image version. Start with standard parameters.
            else:
                new_layer = Layer(1., 0, False)
            version_selected.add_layer(new_layer)

            # Update all layer widgets.
            self.select_version(self.data_object.version_selected)

    def remove_layer(self, layer_index):
        """
        Remove a layer from the selected version. The "remove_layer" method of class "Layer" makes
        sure that the first layer is not removed.

        :param layer_index: Index of the layer to be removed.
        :return: -
        """

        version_selected = self.data_object.versions[self.data_object.version_selected]
        version_selected.remove_layer(layer_index)

        # Update all layer widgets.
        self.select_version(self.data_object.version_selected)

    def accept(self):
        """
        When the user presses the "OK" button, save the currently selected version to the standard
        path and exit.

        :return: -
        """

        self.version_manager_widget.save_version()

        # Terminate the image processor thread.
        self.image_processor.stop()
        self.close()

    def reject(self):
        """
        When the user presses the "Cancel" button, terminate the image processor thread and
        exit without saving anything.

        :return: -
        """

        self.image_processor.stop()
        self.close()


class EmulateStatusBar(object):
    """
    This class is only used for unit testing with the main program below. It emulates the status
    bar of the main GUI by printing the message to standard output.
    """

    def __init__(self):
        super(EmulateStatusBar, self).__init__()
        pass

    @staticmethod
    def print_status_bar_info(message, color):
        """
        Emulate the "write_status_bar" method of the main GUI.

        :param message: Message to be displayed (str).
        :param color: String with the color of the message (e.g. "red").
        :return:
        """

        colors = {}
        colors['red'] = "\033[1;31m"
        colors['blue'] = "\033[1;34m"
        colors['cyan'] = "\033[1;36m"
        colors['green'] = "\033[0;32m"
        colors['reset'] = "\033[0;0m"
        colors['bold'] = "\033[;1m"

        # Add a try-except clause to avoid a program crash if an un-supported color is specified.
        # In this case, just use a standard print.
        try:
            stdout.write(colors[color])
            print(message)
            stdout.write(colors['reset'])
        except:
            print(message)


if __name__ == '__main__':

    input_file_name = "D:\SW-Development\Python\PlanetarySystemStacker\Examples\Moon_2018-03-24\Moon_Tile-024_043939_pss.tiff"
    # Change colors to standard RGB
    input_image = cvtColor(imread(input_file_name, -1), COLOR_BGR2GRAY)

    dummy_status_bar = EmulateStatusBar()
    app = QtWidgets.QApplication(argv)
    window = PostprocEditorWidget(input_image, input_file_name, "_gpp", 1., 0.2,
                                  dummy_status_bar.print_status_bar_info)
    window.show()
    app.exec_()
