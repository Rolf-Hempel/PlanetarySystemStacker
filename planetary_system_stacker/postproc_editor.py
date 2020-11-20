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
from sys import argv, stdout
from time import sleep

from PyQt5 import QtWidgets, QtCore
from cv2 import imread, cvtColor, COLOR_BGR2RGB, GaussianBlur, bilateralFilter, BORDER_DEFAULT
from math import sqrt
from numpy import uint16, float32

from configuration import Configuration, PostprocLayer
from frame_viewer import FrameViewer
from frames import Frames
from miscellaneous import Miscellaneous
from postproc_editor_gui import Ui_postproc_editor
from sharpening_layer_widget import Ui_sharpening_layer_widget
from version_manager_widget import Ui_version_manager_widget


class SharpeningLayerWidget(QtWidgets.QWidget, Ui_sharpening_layer_widget):
    """
    GUI widget to manipulate the parameters of a sharpening layer. An arbitrary number of sharpening
    layers can be defined (up to a maximum set as configuration parameter).
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
        self.horizontalSlider_bi_fraction.valueChanged.connect(self.horizontalSlider_bi_fraction_changed)
        self.lineEdit_bi_fraction.textChanged.connect(self.lineEdit_bi_fraction_changed)
        self.horizontalSlider_bi_range.valueChanged.connect(
            self.horizontalSlider_bi_range_changed)
        self.lineEdit_bi_range.textChanged.connect(self.lineEdit_bi_range_changed)
        self.horizontalSlider_denoise.valueChanged.connect(
            self.horizontalSlider_denoise_changed)
        self.lineEdit_denoise.textChanged.connect(self.lineEdit_denoise_changed)
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
        self.lineEdit_amount.setText("{0:.2f}".format(round(self.layer.amount, 2)))
        self.horizontalSlider_bi_fraction.setValue(
            self.bi_fraction_to_integer(self.layer.bi_fraction))
        self.lineEdit_bi_fraction.setText("{0:.2f}".format(round(self.layer.bi_fraction, 2)))
        self.horizontalSlider_bi_range.setValue(self.bi_range_to_integer(self.layer.bi_range))
        self.lineEdit_bi_range.setText(str(self.layer.bi_range))
        self.horizontalSlider_denoise.setValue(
            self.denoise_to_integer(self.layer.denoise))
        self.lineEdit_denoise.setText("{0:.2f}".format(round(self.layer.denoise, 2)))

        # Temporarily block signals for the luminance checkbox. Otherwise the variable would be
        # switched immediately.
        self.checkBox_luminance.blockSignals(True)
        self.checkBox_luminance.setChecked(self.layer.luminance_only)
        self.checkBox_luminance.blockSignals(False)

    # The following four methods implement the translation between data model values and the
    # (integer) values of the horizontal GUI sliders. In the case of the sharpening amount this
    # translation is non-linear in order to add resolution at small values.
    @staticmethod
    def radius_to_integer(radius):
        return max(min(round(radius * 10.), 99), 1)

    @staticmethod
    def integer_to_radius(int):
        return int / 10.

    @staticmethod
    def amount_to_integer(amount):
        # Below the slider value 20 (amount = 1.) the behaviour is linear, above quadratic.
        if amount > 1.:
            a = 187. / 6400.
            b = 0.15 - 40. * a
            c = 400. * a - 2.
            return (round(-b / (2. * a) + sqrt(b**2 / a**2 / 4. - (c - amount) / a)))
        else:
            return round((amount + 2.) / 0.15)

    @staticmethod
    def integer_to_amount(integer):
        if integer <= 20:
            return -2. + 0.15 * integer
        else:
            a = 187. / 6400.
            b = 0.15 - 40. * a
            c = 400. * a - 2.
            return round(a * integer**2 + b * integer + c, 2)

    @staticmethod
    def bi_fraction_to_integer(bi_fraction):
        return round(bi_fraction * 100)

    @staticmethod
    def integer_to_bi_fraction(integer):
        return integer / 100.

    @staticmethod
    def bi_range_to_integer(bi_range):
        # Below the slider value 20 (amount = 1.) the behaviour is linear, above quadratic.
        if bi_range > 20.:
            a = 113. / 2940.
            b = - 241 / 147.
            c = 900. * a
            return (round(-b / (2. * a) + sqrt(b ** 2 / a ** 2 / 4. - (c - bi_range) / a)))
        else:
            return round(bi_range * 3. / 2.)

    @staticmethod
    def integer_to_bi_range(integer):
        if integer <= 30:
            return round(2. / 3. * integer)
        else:
            a = 113. / 2940.
            b = - 241 / 147.
            c = 900. * a
            return round(a * integer ** 2 + b * integer + c)

    @staticmethod
    def denoise_to_integer(denoise):
        return round(denoise * 100)

    @staticmethod
    def integer_to_denoise(integer):
        return integer / 100.

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

    def horizontalSlider_bi_fraction_changed(self):
        """
        The same as above for the "bi_fraction" parameter.

        :return: -
        """

        self.layer.bi_fraction = self.integer_to_bi_fraction(self.horizontalSlider_bi_fraction.value())
        self.lineEdit_bi_fraction.blockSignals(True)
        self.lineEdit_bi_fraction.setText("{0:.2f}".format(round(self.layer.bi_fraction, 2)))
        self.lineEdit_bi_fraction.blockSignals(False)

    def lineEdit_bi_fraction_changed(self):
        """
        The same as above for the "bi_fraction" parameter.

        :return: -
        """

        try:
            self.layer.bi_fraction = float(self.lineEdit_bi_fraction.text())
            self.horizontalSlider_bi_fraction.blockSignals(True)
            self.horizontalSlider_bi_fraction.setValue(self.bi_fraction_to_integer(self.layer.bi_fraction))
            self.horizontalSlider_bi_fraction.blockSignals(False)
        except:
            pass

    def horizontalSlider_bi_range_changed(self):
        """
        The same as above for the "bi_range" parameter.

        :return: -
        """

        self.layer.bi_range = self.integer_to_bi_range(self.horizontalSlider_bi_range.value())
        self.lineEdit_bi_range.blockSignals(True)
        self.lineEdit_bi_range.setText(str(self.layer.bi_range))
        self.lineEdit_bi_range.blockSignals(False)

    def lineEdit_bi_range_changed(self):
        """
        The same as above for the "bi_range" parameter.

        :return: -
        """

        try:
            self.layer.bi_range = max(0, min(int(round(float(self.lineEdit_bi_range.text()))), 255))
            self.horizontalSlider_bi_range.blockSignals(True)
            self.horizontalSlider_bi_range.setValue(self.bi_range_to_integer(self.layer.bi_range))
            self.horizontalSlider_bi_range.blockSignals(False)
        except:
            pass

    def horizontalSlider_denoise_changed(self):
        """
        The same as above for the "denoise" parameter.

        :return: -
        """

        self.layer.denoise = self.integer_to_denoise(self.horizontalSlider_denoise.value())
        self.lineEdit_denoise.blockSignals(True)
        self.lineEdit_denoise.setText("{0:.2f}".format(round(self.layer.denoise, 2)))
        self.lineEdit_denoise.blockSignals(False)

    def lineEdit_denoise_changed(self):
        """
        The same as above for the "denoise" parameter.

        :return: -
        """

        try:
            self.layer.denoise = float(self.lineEdit_denoise.text())
            self.horizontalSlider_denoise.blockSignals(True)
            self.horizontalSlider_denoise.setValue(self.denoise_to_integer(self.layer.denoise))
            self.horizontalSlider_denoise.blockSignals(False)
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

    def __init__(self, configuration, select_version_callback, parent=None):
        """

        :param configuration: Configuration object with parameters.
        :param select_version_callback: Higher-level method to set the currently selected version.
        :param parent: parent object.
        """

        QtWidgets.QWidget.__init__(self, parent)
        self.setupUi(self)
        self.pss_version = configuration.global_parameters_version
        self.postproc_data_object = configuration.postproc_data_object
        self.postproc_blinking_period = configuration.postproc_blinking_period
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
        self.spinBox_version.setMaximum(configuration.postproc_data_object.number_versions)
        self.spinBox_version.setMinimum(0)
        self.spinBox_compare.setMaximum(configuration.postproc_data_object.number_versions)
        self.spinBox_compare.setMinimum(0)

        # Set the spinbox to the newly created version.
        self.spinBox_version.setValue(self.postproc_data_object.version_selected)

    def select_version(self):
        """
        This method is called when the user changes the setting of the "version selected" spinbox.

        :return: -
        """
        self.postproc_data_object.version_selected = self.spinBox_version.value()
        self.select_version_callback(self.postproc_data_object.version_selected)

    def select_version_compared(self):
        """
        This method is called when the user changes the setting of the "version compared" spinbox.
        The "selected" and "compared" versions are displayed alternately by the blink comparator.
        :return: -
        """

        self.postproc_data_object.version_compared = self.spinBox_compare.value()

    def new_version(self):
        """
        Create a new postprocessing version.
        :return: -
        """

        # Create a Version object by copying the parameters from the selected version.
        new_version = self.postproc_data_object.new_postproc_version_from_existing()

        # If the new version was created from the original image (version 0), add an initial layer.
        if self.postproc_data_object.version_selected == 1:
            new_version.add_postproc_layer(PostprocLayer("Multilevel unsharp masking", 1., 1., 0.,
                                                         20, 0., False))

        # Set the image viewer to the new version, and increase the range of spinboxes to include
        # the new version.
        self.set_photo_signal.emit(self.postproc_data_object.version_selected)
        self.spinBox_version.setMaximum(self.postproc_data_object.number_versions)
        self.spinBox_version.setValue(self.postproc_data_object.number_versions)
        self.spinBox_compare.setMaximum(self.postproc_data_object.number_versions)

    def remove_version(self):
        """
        Remove a postprocessing version.
        :return: -
        """

        # Remove the version from the central data object, and instruct the image viewer to
        # show the previous version.
        self.postproc_data_object.remove_postproc_version(self.postproc_data_object.version_selected)
        self.set_photo_signal.emit(self.postproc_data_object.version_selected)

        # Adjust the spinbox bounds, and set the current version parameters to the new selection.
        self.spinBox_version.setMaximum(self.postproc_data_object.number_versions)

        # Set the spinbox to the version before the deleted one.
        self.spinBox_version.setValue(self.postproc_data_object.version_selected)

        self.spinBox_compare.setMaximum(self.postproc_data_object.number_versions)
        self.spinBox_compare.setValue(
            min(self.spinBox_compare.value(), self.postproc_data_object.number_versions))
        self.select_version_callback(self.postproc_data_object.version_selected)

    def blinking_toggled(self):
        """
        Switch the blink comparator on or off.

        :return: -
        """

        # Toggle the status variable.
        self.postproc_data_object.blinking = not self.postproc_data_object.blinking

        # The blink comparator is switched on.
        if self.postproc_data_object.blinking:
            # Create the blink comparator thread and start it.
            self.blink_comparator = BlinkComparator(self.postproc_data_object,
                                                    self.postproc_blinking_period,
                                                    self.set_photo_signal,
                                                    self.variant_shown_signal)
            self.blink_comparator.setTerminationEnabled(True)

        # The blink comparator is switched off. Set the image viewer to the currently selected
        # image and terminate the separate thread.
        else:
            self.set_photo_signal.emit(self.postproc_data_object.version_selected)
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
        save the result as 16bit png, tiff or fits file at the standard location.

        :return: -
        """

        Frames.save_image(self.postproc_data_object.file_name_processed,
                          self.postproc_data_object.versions[
                              self.postproc_data_object.version_selected].image,
                          color=self.postproc_data_object.color, avoid_overwriting=False,
                          header=self.pss_version)

    def save_version_as(self):
        """
        save the result as 16bit png, tiff or fits file at a location selected by the user.

        :return: -
        """

        options = QtWidgets.QFileDialog.Options()
        filename, extension = QtWidgets.QFileDialog.getSaveFileName(self,
                            "Save result as 16bit png, tiff or fits image",
                            self.postproc_data_object.file_name_processed,
                            "Image Files (*.png *.tiff *.fits)", options=options)

        if filename and extension:
            Frames.save_image(filename,
                              self.postproc_data_object.versions[
                                  self.postproc_data_object.version_selected].image,
                              color=self.postproc_data_object.color, avoid_overwriting=False,
                              header=self.pss_version)


class BlinkComparator(QtCore.QThread):
    """
    This class implements the blink comparator. It shows alternately two image versions.
    """

    def __init__(self, postproc_data_object, postproc_blinking_period, set_photo_signal,
                 variant_shown_signal, parent=None):
        """
        :param postproc_data_object: The central postprocessing data object.
        :param postproc_blinking_period: Time between image switching.
        :param set_photo_signal: Signal which tells the image viewer which image it should show.
        :param variant_shown_signal: Signal which tells the GUI widget which spinbox should be
                                     highlighted.
        :param parent: Parent object
        """

        QtCore.QThread.__init__(self, parent)
        self.postproc_data_object = postproc_data_object
        self.postproc_blinking_period = postproc_blinking_period
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
        while self.postproc_data_object.blinking:
            # Show the "selected" version.
            if show_selected_version:
                self.set_photo_signal.emit(self.postproc_data_object.version_selected)
                self.variant_shown_signal.emit(True, False)
            # Show the "compared" version.
            else:
                self.set_photo_signal.emit(self.postproc_data_object.version_compared)
                self.variant_shown_signal.emit(False, True)

            # Toggle back and forth between first and second image version.
            show_selected_version = not show_selected_version

            # Sleep time inserted to limit CPU consumption by idle looping.
            sleep(self.postproc_blinking_period)

    def stop(self):
        """
        Terminate the blink comparator thread.

        :return: -
        """

        self.variant_shown_signal.emit(False, False)
        self.terminate()


class ImageProcessor(QtCore.QThread):
    """
    This class implements the asynchronous computation of new image versions.
    """

    # The set_photo_signal tells the image viewer which image it should show.
    set_photo_signal = QtCore.pyqtSignal(int)
    # The set_status_bar_signal triggers the display of a (colored) message in the main GUI's
    # status bar.
    set_status_bar_signal = QtCore.pyqtSignal(str, str)

    def __init__(self, configuration, parent=None):
        """
        Whenever the parameters of the current version change, compute a new sharpened image.

        :param configuration: Data object with configuration parameters
        """

        QtCore.QThread.__init__(self, parent)
        self.postproc_data_object = configuration.postproc_data_object
        self.postproc_idle_loop_time = configuration.postproc_idle_loop_time

        # Change the main GUI's status bar to show that a computation is going on.
        self.set_status_bar_signal.emit(
            "Processing " + self.postproc_data_object.file_name_original +
            ", busy computing a new postprocessing image.", "black")

        # Initialize images for all versions using the current layer data.
        self.last_version_layers = []
        for index, version in enumerate(self.postproc_data_object.versions):
            self.last_version_layers.append(deepcopy(version.layers))
            version.image = ImageProcessor.recompute_selected_version(
                self.postproc_data_object.image_original, version.layers)

        # Reset the status bar to its idle state.
        self.set_status_bar_signal.emit(
                "Processing " + self.postproc_data_object.file_name_original + ", postprocessing.",
                "black")

        # Remember the last version (and the corresponding layer parameters) shown in the image
        # viewer. As soon as this index changes, a new image is displayed.
        self.last_version_selected = -1

        # Initialize the data objects holding the currently active version.
        self.version_selected = None
        self.layers_selected = None

        # Remember that a new image must be computed because the ImageProcessor is just initialized.
        self.initial_state = True

        self.start()

    def run(self):
        while True:
            # To avoid chasing a moving target, copy the parameters of the currently active version
            self.version_selected = self.postproc_data_object.version_selected
            self.layers_selected = self.postproc_data_object.versions[
                                                self.version_selected].layers

            # Compare the currently active version with the last one for which an image was
            # computed. If there is a difference, start a new computation.
            if self.new_computation_required() and self.version_selected:

                # Change the main GUI's status bar to show that a computation is going on.
                self.set_status_bar_signal.emit(
                    "Processing " + self.postproc_data_object.file_name_original +
                    ", busy computing a new postprocessing image.", "black")

                # Perform the new computation.
                self.postproc_data_object.versions[
                    self.version_selected].image = ImageProcessor.recompute_selected_version(
                        self.postproc_data_object.image_original, self.layers_selected)

                # Reset the status bar to its idle state.
                self.set_status_bar_signal.emit(
                    "Processing " + self.postproc_data_object.file_name_original +
                    ", postprocessing.", "black")

                # Show the new image in the image viewer, and remember its parameters.
                self.set_photo_signal.emit(self.version_selected)
                self.last_version_selected = self.version_selected

            elif self.version_selected != self.last_version_selected:
                # Show the new image in the image viewer, and remember its parameters.
                self.set_photo_signal.emit(self.version_selected)
                self.last_version_selected = self.version_selected

            # Idle loop before doing the next check for updates.
            else:
                sleep(self.postproc_idle_loop_time)

    def new_computation_required(self):
        """
        Decide if a new computation is required.

        :return: True, if the current version parameters have changed. False, otherwise.
        """

        # When the ImageProcessor is just initialized, a new image must be computed.
        if self.initial_state:
            self.initial_state = False
            return True

        # Check if an additional version was created. Copy its layer info for later checks
        # for changes.
        if self.version_selected >= len(self.last_version_layers):
            self.last_version_layers.append(deepcopy(self.postproc_data_object.versions[
                                                         self.version_selected].layers))
            return True

        # If the selected version is already known, check if the number of layers has changed.
        if len(self.last_version_layers[self.version_selected]) != len(self.layers_selected):
            self.last_version_layers[self.version_selected] = deepcopy(self.layers_selected)
            return True

        # For all layers check if a parameter has changed.
        for last_layer, layer_selected in zip(self.last_version_layers[self.version_selected],
                                              self.layers_selected):
            if last_layer.postproc_method != layer_selected.postproc_method or \
                    last_layer.radius != layer_selected.radius or \
                    last_layer.amount != layer_selected.amount or \
                    last_layer.bi_fraction != layer_selected.bi_fraction or \
                    last_layer.bi_range != layer_selected.bi_range or \
                    last_layer.denoise != layer_selected.denoise or \
                    last_layer.luminance_only != layer_selected.luminance_only:
                self.last_version_layers[self.version_selected] = deepcopy(self.layers_selected)
                return True

        # No change detected.
        return False

    @staticmethod
    def recompute_selected_version(input_image, layers):
        """
        Do the actual computation. Starting from the original image, for each sharpening layer
        apply a Gaussian filter using the layer's parameters. Store the result in the central
        data object for the selected version.

        :return: -
        """

        # Divide the input image in components corresponding to the sharpening layers. The input
        # image is still the sum of all those components.
        image_layer_components = []
        previous_blurred_image = input_image.astype(float32)
        for layer in layers:
            if layer.bi_fraction == 1.:
                image_blurred = bilateralFilter(previous_blurred_image, 0, layer.bi_range * 256.,
                                                layer.radius/3., borderType=BORDER_DEFAULT)
            elif layer.bi_fraction == 0.:
                image_blurred = GaussianBlur(previous_blurred_image, (0, 0), layer.radius/3.,
                                             borderType=BORDER_DEFAULT)
            else:
                image_blurred = bilateralFilter(previous_blurred_image, 0, layer.bi_range * 256.,
                                                layer.radius / 3.,
                                                borderType=BORDER_DEFAULT) * layer.bi_fraction + \
                                GaussianBlur(previous_blurred_image, (0, 0), layer.radius / 3.,
                                             borderType=BORDER_DEFAULT) * (1. - layer.bi_fraction)
            image_layer_components.append(previous_blurred_image - image_blurred)
            previous_blurred_image = image_blurred

        # Build the sharpened image as a weighted sum of the layer components.
        new_image = previous_blurred_image
        for layer, image_layer_component in zip(layers, image_layer_components):
            new_image += image_layer_component * layer.amount

        # Clip pixels out of range and convert the processed image to 16bit unsigned int.
        return new_image.clip(min=0., max=65535.).astype(uint16)

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
    a variable number of sharpening layers. A blink comparator allows comparing any two versions.
    """

    def __init__(self, configuration, image_original, name_original, set_status_bar_callback,
                 signal_save_postprocessed_image):
        """

        :param configuration: Configuration object with parameters.
        :param image_original: Original image (16bit) holding the input for postprocessing.
        :param name_original: Path name of the original image.
        :param set_status_bar_callback: Call-back function to update the main GUI's status bar.
        :param signal_save_postprocessed_image: Signal to be issued when the postprocessing
                                                widget closes. None if no signal is to be issued.
        """

        super(PostprocEditorWidget, self).__init__()
        self.setupUi(self)

        self.configuration = configuration
        self.postproc_data_object = self.configuration.postproc_data_object
        self.postproc_data_object.set_postproc_input_image(image_original, name_original,
                                                self.configuration.global_parameters_image_format)
        self.signal_save_postprocessed_image = signal_save_postprocessed_image

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.pushButton_add_layer.clicked.connect(self.add_layer)

        # Initialize list of sharpening layer widgets, and set the maximal number of layers.
        self.sharpening_layer_widgets = []
        self.max_layers = self.configuration.postproc_max_layers

        self.label_message.setText("Create sharp image versions using up to "
                                   + str(self.max_layers) + " correction layers."
                                   " Adjust layer parameters as required.")
        self.label_message.setStyleSheet('color: red')

        # Start the frame viewer.
        self.frame_viewer = FrameViewer()
        self.frame_viewer.setObjectName("framewiever")
        self.gridLayout.addWidget(self.frame_viewer, 0, 0, 3, 1)

        # Initialize a vertical spacer used to fill the lower part of the sharpening widget scroll
        # area.
        self.spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum,
                                            QtWidgets.QSizePolicy.Expanding)

        # Create the version manager and pass it the "select_version" callback function.
        self.version_manager_widget = VersionManagerWidget(self.configuration, self.select_version)
        self.gridLayout.addWidget(self.version_manager_widget, 2, 1, 1, 1)

        # The "set_photo_signal" from the VersionManagerWidget is not passed to the image viewer
        # directly. (The image viewer does not accept signals.) Instead, it calls the "select_image"
        # method below which in turn invokes the image viewer.
        self.version_manager_widget.set_photo_signal.connect(self.select_image)

        # Select the initial current version.
        self.select_version(self.postproc_data_object.version_selected)

        # Start the image processor thread, and connect its signals.
        self.image_processor = ImageProcessor(self.configuration)
        self.image_processor.setTerminationEnabled(True)
        self.image_processor.set_photo_signal.connect(self.select_image)
        self.image_processor.set_status_bar_signal.connect(set_status_bar_callback)

    def select_version(self, version_index):
        """
        Select a new current version and update the scroll area with all layer widgets.

        :param version_index: Index of the version selected in version list of the central data
                              object.
        :return: -
        """

        version_selected = self.postproc_data_object.versions[version_index]

        # Remove all existing layer widgets and the lower vertical spacer.
        if self.sharpening_layer_widgets:
            for layer_widget in self.sharpening_layer_widgets:
                self.verticalLayout.removeWidget(layer_widget)
                layer_widget.deleteLater()
                del layer_widget
        self.sharpening_layer_widgets = []
        self.verticalLayout.removeItem(self.spacerItem)

        # Create new layer widgets for all active layers and put them into the scroll area.
        for layer_index, layer in enumerate(version_selected.layers):
            sharpening_layer_widget = SharpeningLayerWidget(layer_index, self.remove_layer)
            sharpening_layer_widget.set_values(layer)
            self.verticalLayout.addWidget(sharpening_layer_widget)
            self.sharpening_layer_widgets.append(sharpening_layer_widget)

        # At the end of the scroll area, add a vertical spacer.
        self.verticalLayout.addItem(self.spacerItem)

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

        self.frame_viewer.setPhoto(self.postproc_data_object.versions[version_index].image)

    def add_layer(self):
        """
        This method is called when the "Add postprocessing layer" button is pressed. Add a layer
        to the currently selected postprocessing version.

        :return:-
        """

        version_selected = self.postproc_data_object.versions[
            self.postproc_data_object.version_selected]
        num_layers_current = version_selected.number_layers

        # The "zero" layer is reserved for the original image, so for this version no additional
        # layer can be allocated.

        # The first layer is initialized with a fixed parameter set (see "init" method of DataObject
        # class).
        if self.postproc_data_object.version_selected and num_layers_current < self.max_layers:

            # This version has already at least one layer. Initialize the radius parameter of the
            # new layer to 1.5 times the radius of the previous one.
            if num_layers_current:
                previous_layer = version_selected.layers[num_layers_current - 1]
                new_layer = PostprocLayer(previous_layer.postproc_method,
                                          round(1.5 * previous_layer.radius, 1), 1.,
                                          previous_layer.bi_fraction,
                                          previous_layer.bi_range, 0.,
                                          previous_layer.luminance_only)

            # This is the first layer for this image version. Start with standard parameters.
            else:
                new_layer = PostprocLayer("Multilevel unsharp masking", 1., 1., 0., 20, 0., False)
            version_selected.add_postproc_layer(new_layer)

            # Update all layer widgets.
            self.select_version(self.postproc_data_object.version_selected)

    def remove_layer(self, layer_index):
        """
        Remove a layer from the selected version. The "remove_layer" method of class "Layer" makes
        sure that the first layer is not removed.

        :param layer_index: Index of the layer to be removed.
        :return: -
        """

        version_selected = self.postproc_data_object.versions[
            self.postproc_data_object.version_selected]
        version_selected.remove_postproc_layer(layer_index)

        # Update all layer widgets.
        self.select_version(self.postproc_data_object.version_selected)

    def accept(self):
        """
        When the user presses the "OK" button, save the currently selected version to the standard
        path and exit.

        :return: -
        """

        self.version_manager_widget.save_version()

        # Terminate the image processor thread.
        self.image_processor.stop()
        self.configuration.write_config()
        if self.signal_save_postprocessed_image:
            self.signal_save_postprocessed_image.emit(self.postproc_data_object.versions[
                self.postproc_data_object.version_selected].image)
        self.close()

    def reject(self):
        """
        When the user presses the "Cancel" button, terminate the image processor thread and
        exit without saving anything.

        :return: -
        """

        self.image_processor.stop()
        if self.signal_save_postprocessed_image:
            self.signal_save_postprocessed_image.emit(None)
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
    input_image = cvtColor(imread(input_file_name, -1), COLOR_BGR2RGB)

    configuration = Configuration()
    configuration.initialize_configuration()
    dummy_status_bar = EmulateStatusBar()
    app = QtWidgets.QApplication(argv)
    window = PostprocEditorWidget(configuration, input_image, input_file_name,
                                  dummy_status_bar.print_status_bar_info, None)
    window.show()
    app.exec_()
