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

from copy import copy, deepcopy
from pathlib import Path
from sys import argv, stdout
from time import sleep

from PyQt5 import QtWidgets, QtCore
from cv2 import imread, cvtColor, COLOR_BGR2RGB, GaussianBlur, bilateralFilter, BORDER_DEFAULT,\
    COLOR_BGR2HSV, COLOR_HSV2BGR
from math import sqrt
from numpy import uint8, uint16, float32

from configuration import Configuration, PostprocLayer
from exceptions import InternalError
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
    signal_toggle_luminance_other_layers = QtCore.pyqtSignal()

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
        # Below the slider value 50 (bi_range = 13.) the behaviour is linear, above quadratic.
        if bi_range > 13.:
            c = 0.009022222222222221
            g = 35.59113300492611
            h = -1233.2712514256596
            return round(g + sqrt(h + bi_range / c))
        else:
            b = 0.26
            return round(bi_range / b)

    @staticmethod
    def integer_to_bi_range(integer):
        if integer <= 50:
            b = 0.26
            return round(b * integer, 1)
        else:
            c = 0.009022222222222221
            e = -0.6422222222222221
            f = 22.555555555555554
            return round(c * integer ** 2 + e * integer + f, 1)

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
        The checkbox which tells if the sharpening should affect the luminance channel only has
        changed its state.

        :return: -
        """

        self.layer.luminance_only = not self.layer.luminance_only

        # Signal the other layers to change the flag as well.
        self.signal_toggle_luminance_other_layers.emit()

    @QtCore.pyqtSlot()
    def checkBox_luminance_toggled_no_signal(self):
        """
        The checkbox which tells if the sharpening should affect the luminance channel only has
        changed its state. This version of the method is invoked by another layer which has changed
        the toggle state. In this case, not only toggle the state variable, but also toggle the
        checkbox widget (because this was not done by the user). To avoid an avalanche, do not emit
        a signal.

        :return: -
        """

        # Block the signals of this widget to avoid a feedback loop.
        self.checkBox_luminance.blockSignals(True)
        self.checkBox_luminance.setChecked(not self.checkBox_luminance.isChecked())
        self.checkBox_luminance.blockSignals(False)
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
        self.configuration = configuration
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
                                    self.configuration.postproc_bi_range_standard, 0., False))

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
    # The set_shift_display_signal tells the PostprocEditorWidget to update the display of RGB
    # shift values.
    set_shift_display_signal = QtCore.pyqtSignal()
    # The set_status_bar_signal triggers the display of a (colored) message in the main GUI's
    # status bar.
    set_status_bar_signal = QtCore.pyqtSignal(str, str)

    def __init__(self, configuration, parent=None):
        """
        Whenever the parameters of the current version change, compute a new sharpened image.

        :param configuration: Data object with configuration parameters
        """

        QtCore.QThread.__init__(self, parent)
        self.configuration = configuration
        self.postproc_data_object = self.configuration.postproc_data_object
        self.postproc_idle_loop_time = self.configuration.postproc_idle_loop_time

        # Extract the file name from its path.
        self.file_name = Path(self.postproc_data_object.file_name_original).name

        # Do the computations in float32 to avoid clipping effects. If the input image is color,
        # also create a version for "luminance only" computations.
        self.image_original = self.postproc_data_object.image_original.astype(float32)
        self.color = len(self.image_original.shape) == 3
        if self.color:
            self.image_original_hsv = cvtColor(self.image_original, COLOR_BGR2HSV)

        # Initialize list of auto-rgb-aligned input images for all resolution levels.
        self.auto_rgb_aligned_images_original = [None, None, None]
        if self.color:
            self.auto_rgb_aligned_images_original_hsv = [None, None, None]
        self.auto_rgb_shifts_red = [None, None, None]
        self.auto_rgb_shifts_blue = [None, None, None]

        # Change the main GUI's status bar to show that a computation is going on.
        self.set_status_bar_signal.emit("Postprocessing " + self.file_name +
            ", busy computing initial images for all versions.", "black")

        # Initialize images for all versions using the current layer data.
        self.last_version_layers = []
        # self.last_version_rgb_aligned = [None]*self.configuration.postproc_max_versions
        for version in self.postproc_data_object.versions:
            # For every version, keep a copy of the current layer parameters for later checks for
            # changes. Also, remember if for this version RGB alignment was active.
            self.last_version_layers.append(deepcopy(version.layers))
            version.last_rgb_automatic = version.rgb_automatic

            # If automatic RGB alignment is selected for this version, compute the shifted
            # original image for this version's resolution if not yet available.
            if version.rgb_automatic:
                if self.auto_rgb_aligned_images_original[version.rgb_resolution_index] is None:
                    try:
                        self.auto_rgb_aligned_images_original[version.rgb_resolution_index], \
                        self.auto_rgb_shifts_red[version.rgb_resolution_index], \
                        self.auto_rgb_shifts_blue[
                            version.rgb_resolution_index] = Miscellaneous.auto_rgb_align(
                            self.image_original, self.configuration.postproc_max_shift,
                            interpolation_factor=[1, 2, 4][version.rgb_resolution_index],
                            blur_strength=version.rgb_gauss_width)
                        if self.color:
                            self.auto_rgb_aligned_images_original_hsv[
                                version.rgb_resolution_index] = cvtColor(
                                self.auto_rgb_aligned_images_original[version.rgb_resolution_index],
                                COLOR_BGR2HSV)
                    except:
                        self.auto_rgb_aligned_images_original[
                            version.rgb_resolution_index] = self.image_original
                        self.auto_rgb_shifts_red[version.rgb_resolution_index] = (0., 0.)
                        self.auto_rgb_shifts_blue[version.rgb_resolution_index] = (0., 0.)
                        if self.color:
                            self.auto_rgb_aligned_images_original_hsv[
                                version.rgb_resolution_index] = self.image_original_hsv

                version.image = Miscellaneous.post_process(
                    self.auto_rgb_aligned_images_original[version.rgb_resolution_index],
                    version.layers)
            else:
                version.image = Miscellaneous.post_process(self.image_original, version.layers)

        # Initialize lists for intermediate results (to speed up image updates). The efficient
        # reuse of intermediate results is only possible if there is enough RAM.
        self.reset_intermediate_images()
        self.ram_sufficient = True

        # Reset the status bar to its idle state.
        self.set_status_bar_signal.emit("Postprocessing " + self.file_name,
                "black")

        # Remember the last version (and the corresponding layer parameters) shown in the image
        # viewer. As soon as this index changes, a new image is displayed.
        self.last_version_selected = -1

        # Initialize the data objects holding the currently active and the previous version.
        self.version_selected = None
        self.layers_selected = None

        self.start()

    def reset_intermediate_images(self):
        """
        Initialize all intermediate image versions, so that they will be re-computed next time.

        :return: -
        """

        self.layer_input = [None] * (self.configuration.postproc_max_layers + 1)
        self.layer_gauss = [None] * (self.configuration.postproc_max_layers)
        self.layer_bilateral = [None] * (self.configuration.postproc_max_layers)
        self.layer_denoised = [None] * (self.configuration.postproc_max_layers)

    def run(self):
        while True:
            # To avoid chasing a moving target, copy the parameters of the currently active version
            self.version_selected = self.postproc_data_object.version_selected
            postproc_version = deepcopy(self.postproc_data_object.versions[self.version_selected])
            self.layers_selected = postproc_version.layers
            self.rgb_automatic = postproc_version.rgb_automatic
            self.rgb_resolution_index = postproc_version.rgb_resolution_index
            # print ("resolution index: " + str(self.rgb_resolution_index))
            self.rgb_gauss_width = postproc_version.rgb_gauss_width
            self.shift_red = postproc_version.shift_red
            self.shift_blue = postproc_version.shift_blue
            self.correction_red = postproc_version.correction_red
            self.correction_blue = postproc_version.correction_blue
            self.correction_red_saved = postproc_version.correction_red_saved
            self.correction_blue_saved = postproc_version.correction_blue_saved

            compute_new_image = False
            shift_image = False

            # If the RGB auto-alignment checkbox was changed since the last image was computed for
            # this version, invalidate all intermediate results for this version.
            if self.rgb_automatic != self.postproc_data_object.versions[
                self.version_selected].last_rgb_automatic:
                self.postproc_data_object.versions[
                    self.version_selected].last_rgb_automatic = self.rgb_automatic
                self.reset_intermediate_images()
                compute_new_image = True

            # If automatic RGB alignment is on, check if the shifted image for the current
            # resolution has been computed already. Otherwise, compute it now.
            if self.rgb_automatic:
                if self.auto_rgb_aligned_images_original[
                    self.rgb_resolution_index] is None:
                    # Change the main GUI's status bar to show that a computation is going on.
                    self.set_status_bar_signal.emit("Postprocessing " + self.file_name +
                                                    ", busy computing a new image.", "black")
                    try:
                        self.auto_rgb_aligned_images_original[self.rgb_resolution_index], \
                        self.auto_rgb_shifts_red[self.rgb_resolution_index], \
                        self.auto_rgb_shifts_blue[
                            self.rgb_resolution_index] = Miscellaneous.auto_rgb_align(
                            self.image_original, self.configuration.postproc_max_shift,
                            interpolation_factor=[1, 2, 4][self.rgb_resolution_index],
                            blur_strength=self.rgb_gauss_width)
                        if self.color:
                            self.auto_rgb_aligned_images_original_hsv[
                                self.rgb_resolution_index] = cvtColor(
                                self.auto_rgb_aligned_images_original[self.rgb_resolution_index],
                                COLOR_BGR2HSV)
                    except:
                        self.auto_rgb_aligned_images_original[
                            self.rgb_resolution_index] = self.image_original
                        self.auto_rgb_shifts_red[self.rgb_resolution_index] = (0., 0.)
                        self.auto_rgb_shifts_blue[self.rgb_resolution_index] = (0., 0.)
                        if self.color:
                            self.auto_rgb_aligned_images_original_hsv[
                                self.rgb_resolution_index] = self.image_original_hsv
                    # Reset the status bar to its idle state.
                    self.set_status_bar_signal.emit("Postprocessing " + self.file_name, "black")

                    # Since the auto-shifted image is new, the postprocessing pipeline must be
                    # applied.
                    compute_new_image = True

                # Set the processing pipeline input to the RGB aligned original image.
                self.input_image = self.auto_rgb_aligned_images_original[self.rgb_resolution_index]
                if self.color:
                    self.input_image_hsv = self.auto_rgb_aligned_images_original_hsv[
                            self.rgb_resolution_index]

                # Set the RGB shifts for this version to the values computed automatically.
                self.postproc_data_object.versions[self.version_selected].shift_red = \
                    self.auto_rgb_shifts_red[self.rgb_resolution_index]
                self.postproc_data_object.versions[self.version_selected].shift_blue = \
                    self.auto_rgb_shifts_blue[self.rgb_resolution_index]

            # In correction mode the wavelets are computed only once, and then only the shift
            # corrections are applied.
            elif postproc_version.rgb_correction_mode:

                # If the uncorrected image for this resolution index is not available, it must be
                # computed. Try if an image for this version has been computed. If not, this version
                # is new. In that case process the wavelets first (only applying the accumulated
                # shifts), and do the shift corrections in the next pass.
                if postproc_version.images_uncorrected[self.rgb_resolution_index] is None:
                    if postproc_version.image is None or postproc_version.image.dtype == uint8:
                        self.input_image = Miscellaneous.shift_colors(self.image_original,
                                                                      postproc_version.shift_red,
                                                                      postproc_version.shift_blue)
                        if self.color:
                            if postproc_version.shift_red != (
                            0., 0.) or postproc_version.shift_blue != (0., 0.):
                                self.input_image_hsv = cvtColor(self.input_image, COLOR_BGR2HSV)
                            else:
                                self.input_image_hsv = self.image_original_hsv

                        # self.postproc_data_object.versions[self.version_selected].shift_red = \
                        #     self.postproc_data_object.versions[self.version_selected].shift_blue = (0., 0.)
                        self.reset_intermediate_images()
                        compute_new_image = True

                    # The postprocessing pipeline has been applied to this version before, set the
                    # input for this resolution index of the correction mode. The image is
                    # blown up by the resolution factor to show shifts in detail.
                    else:
                        # In the first pass, convert the input image to 8bit unsigned int (to speed
                        # up the image viewer). Store it as resolution version 0 (1 Pixel, i.e. no
                        # change in resolution).
                        if self.postproc_data_object.versions[
                            self.version_selected].images_uncorrected[0] is None:
                            self.postproc_data_object.versions[
                                self.version_selected].images_uncorrected[0] = (
                                    postproc_version.image / 256.).astype(
                                uint8)

                        # If the resolution selected is not 1 pixel, interpolate the 1 pixel image
                        # to the required resolution and store the result for later reuse.
                        if self.rgb_resolution_index > 0:
                            self.postproc_data_object.versions[
                                self.version_selected].images_uncorrected[
                                self.rgb_resolution_index] = \
                                Miscellaneous.shift_colors(self.postproc_data_object.versions[
                                    self.version_selected].images_uncorrected[0], (0., 0.), (0., 0.),
                                    interpolate_input=[1, 2, 4][self.rgb_resolution_index])
                        shift_image = True

                # The image without shift correction is available for the required resolution. Check
                # if the shift correction has changed. If so, apply the correction. Otherwise, leave
                # the image unchanged.
                elif self.correction_red != self.correction_red_saved or self.correction_blue != self.correction_blue_saved:
                    # print ("correction_red: " + str(self.correction_red) + ", correction_red_saved: " +
                    #        str(self.correction_red_saved)+ ", correction_blue: " + str(self.correction_blue) +
                    #        ", correction_blue_saved: " + str(self.correction_blue_saved))
                    # print ("red: " + str(self.correction_red != self.correction_red_saved))
                    # print("blue: " + str(self.correction_blue != self.correction_blue_saved))
                    self.postproc_data_object.versions[self.version_selected].correction_red_saved = self.correction_red
                    self.postproc_data_object.versions[self.version_selected].correction_blue_saved = self.correction_blue
                    shift_image = True

            # Wavelet mode, RGB auto-alignment off: Set the input image for the processing pipeline to the original image.
            else:
                self.input_image = self.image_original
                if self.color:
                    self.input_image_hsv = self.image_original_hsv

            self.set_shift_display_signal.emit()

            # print ("shift_image: " + str(shift_image) + ", compute_new_image: " + str(compute_new_image))
            # The image has already passed the postprocessing pipeline (as contained in
            # postproc_version.images_uncorrected). Only the correction shift is to be applied.
            if shift_image:
                interpolation_factor = [1, 2, 4][self.rgb_resolution_index]
                self.postproc_data_object.versions[self.version_selected].image = \
                    Miscellaneous.shift_colors(
                        self.postproc_data_object.versions[
                            self.version_selected].images_uncorrected[self.rgb_resolution_index],
                        (self.correction_red[0] * interpolation_factor,
                         self.correction_red[1] * interpolation_factor),
                        (self.correction_blue[0] * interpolation_factor,
                         self.correction_blue[1] * interpolation_factor))
                # Show the new image in the image viewer, and remember its parameters.
                self.set_photo_signal.emit(self.version_selected)
                self.last_version_selected = self.version_selected

            # A new image must be computed for this version. Special case self.version_selected = 0:
            # For the original image no layers are applied. But if the RGB automatic checkbox was
            # changed, the image must be set according to the new shift status. The shift was
            # applied in self.input_image above.
            elif compute_new_image and not self.version_selected:
                self.postproc_data_object.versions[0].image = self.input_image
                # Show the new image in the image viewer, and remember its parameters.
                self.set_photo_signal.emit(self.version_selected)
                self.last_version_selected = self.version_selected

            # General case for new image computation: Either it was decided above that a new
            # computation is required, or the test "new_computation_required" for changes in the
            # correction layers is performed.
            elif compute_new_image or self.new_computation_required(
                    self.version_selected != self.last_version_selected):
                # Change the main GUI's status bar to show that a computation is going on.
                self.set_status_bar_signal.emit("Postprocessing " + self.file_name +
                                                ", busy computing a new image.", "black")

                # Perform the new computation. Try to store intermediate results. If it fails
                # because there is not enough RAM, switch to direct computation.
                if self.ram_sufficient:
                    try:
                        self.postproc_data_object.versions[
                            self.version_selected].image = self.recompute_selected_version(
                                                            self.layers_selected)
                    except:
                        self.ram_sufficient = False
                        self.postproc_data_object.versions[
                            self.version_selected].image = Miscellaneous.post_process(
                                                            self.input_image, self.layers_selected)
                else:
                    self.postproc_data_object.versions[
                        self.version_selected].image = Miscellaneous.post_process(
                        self.input_image, self.layers_selected)

                # Reset the status bar to its idle state.
                self.set_status_bar_signal.emit("Postprocessing " + self.file_name, "black")

                # Show the new image in the image viewer, and remember its parameters.
                self.set_photo_signal.emit(self.version_selected)
                self.last_version_selected = self.version_selected

            # Neither the shift nor parameters have changed. If the version selection has changed,
            # display the image stored with the version selected.
            elif self.version_selected != self.last_version_selected:
                # Show the new image in the image viewer, and remember its parameters.
                self.set_photo_signal.emit(self.version_selected)
                self.last_version_selected = self.version_selected

            # Idle loop before doing the next check for updates.
            else:
                sleep(self.postproc_idle_loop_time)

    def new_computation_required(self, version_has_changed):
        """
        Decide if a new computation is required. Invalidate only those intermediate image
        components which are affected by the parameter change. This results in a substantial
        performance improvement.

        :param version_has_changed: True, if the version has changed, so all intermediate results
                                    must be invalidated.
        :return: True, if the current version parameters have changed. False, otherwise.
        """

        # Initialize the flag which remembers if a change was found. Do not return on the first
        # detection in order to invalidate all affected intermediate results.
        change_detected = False

        # If a new version is selected, reset images with intermediate results.
        if version_has_changed:
            self.reset_intermediate_images()
            change_detected = True

        # Check if an additional version was created. Copy its layer info for later checks
        # for changes. The intermediate results are reset already because the version number must
        # have changed.
        if self.version_selected >= len(self.last_version_layers):
            self.last_version_layers.append(deepcopy(self.postproc_data_object.versions[
                                                         self.version_selected].layers))
            return change_detected

        # If the selected version is already known, check if the number of layers has changed.
        if len(self.last_version_layers[self.version_selected]) != len(self.layers_selected):
            self.last_version_layers[self.version_selected] = deepcopy(self.layers_selected)
            self.reset_intermediate_images()
            change_detected = True
            return change_detected

        # For color images, check if the "luminance only" parameter has changed. In this case all
        # layers are invalidated completely. Remember that the "luminance only" parameter is the
        # same on all layers for any given version, and that for version 0 (original image) no
        # layers are defined (and therefore there is no parameter "luminance_only").
        if self.color and self.version_selected and \
                self.last_version_layers[self.version_selected][0].luminance_only != \
                self.layers_selected[0].luminance_only:
            self.reset_intermediate_images()
            change_detected = True

        # The version is the same as before, it has the same number of layers, and the luminance
        # parameter has not changed. Now go through all layers and invalidate those intermediate
        # results individually which need to be recomputed.
        else:
            for layer_index, (last_layer, layer_selected) in enumerate(
                    zip(self.last_version_layers[self.version_selected], self.layers_selected)):
                # The "amount" parameter has changed. No intermediate results need to be invalidated
                # because the layer's contribution to the image is computed by multiplying the
                # denoised component with the amount parameter.
                if last_layer.amount != layer_selected.amount:
                    last_layer.amount = layer_selected.amount
                    change_detected = True

                # If the radius has changed, all filters at this level and the input for the next
                # level have to be recomputed.
                if  last_layer.radius != layer_selected.radius:
                    self.layer_gauss[layer_index] = self.layer_bilateral[layer_index] = \
                        self.layer_denoised[layer_index] = self.layer_input[layer_index + 1] = None
                    change_detected = True

                # If the bi_fraction has changed, the input for the next layer and the denoised
                # component on this layer must be recomputed.
                if last_layer.bi_fraction != layer_selected.bi_fraction:
                    self.layer_denoised[layer_index] = self.layer_input[layer_index + 1] = None
                    change_detected = True

                # If the bi_range parameter has changed, the bilateral filter is invalidated.
                if last_layer.bi_range != layer_selected.bi_range:
                    self.layer_bilateral[layer_index] = None
                    change_detected = True
                    # If the bi_fraction parameter is not zero, the change propagates to the
                    # denoised component and the input for the next level.
                    if abs(layer_selected.bi_fraction) > 1.e-5:
                        self.layer_denoised[layer_index] = self.layer_input[layer_index + 1] = None

                # If the denoise parameter has changed, the denoised component for this layer is
                # invalidated.
                if last_layer.denoise != layer_selected.denoise:
                    self.layer_denoised[layer_index] = None
                    change_detected = True

        # Remember the current parameter settings to compare with new paraemters next time.
        self.last_version_layers[self.version_selected] = deepcopy(self.layers_selected)
        return change_detected

    def recompute_selected_version(self, layers):
        """
        Do the actual computation. Starting from the original image, for each sharpening layer
        apply a Gaussian filter using the layer's parameters. Store the result in the central
        data object for the selected version.

        :return: -
        """

        # Check if the original image is selected (version 0). In this case return the (potentially
        # RGB-shifted) original iamge.
        if not layers:
            return self.input_image.astype(uint16)

        for layer_index, layer in enumerate(layers):
            if self.layer_input[layer_index] is None:
                # For layers > 0, the layer_input must have been computed on the previous layer.
                if layer_index:
                    raise InternalError("Layer input image is None for layer > 0")
                # On layer 0, the original image is taken as layer input.
                else:
                    # For color input and luminance_only: extract the luminance channel.
                    if self.color and layer.luminance_only:
                        self.layer_input[layer_index] = self.input_image_hsv[:, :, 2]
                    else:
                        self.layer_input[layer_index] = self.input_image

            # Bilateral filter is needed:
            if abs(layer.bi_fraction) > 1.e-5:
                # Filter must be recomputed.
                if self.layer_bilateral[layer_index] is None:
                    self.layer_bilateral[layer_index] = bilateralFilter(self.layer_input[layer_index],
                        0, layer.bi_range * 256., layer.radius/3., borderType=BORDER_DEFAULT)
            # Gaussian filter is needed:
            if abs(layer.bi_fraction - 1.) > 1.e-5:
                # Filter must be recomputed.
                if self.layer_gauss[layer_index] is None:
                    self.layer_gauss[layer_index] = GaussianBlur(self.layer_input[layer_index], (0, 0),
                        layer.radius/3., borderType=BORDER_DEFAULT)

            # Compute the input image for the next layer.
            if self.layer_input[layer_index + 1] is None:
                # Case bilateral only.
                if abs(layer.bi_fraction - 1.) <= 1.e-5:
                    self.layer_input[layer_index + 1] = self.layer_bilateral[layer_index]
                # Case Gaussian only.
                elif abs(layer.bi_fraction) <= 1.e-5:
                    self.layer_input[layer_index + 1] = self.layer_gauss[layer_index]
                # Mixed case.
                else:
                    self.layer_input[layer_index + 1] = self.layer_bilateral[layer_index] * layer.bi_fraction + \
                                                        self.layer_gauss[layer_index] * (1. - layer.bi_fraction)

            # A new denoised layer component must be computed.
            if self.layer_denoised[layer_index] is None:
                layer_component_before_denoise = self.layer_input[layer_index] - self.layer_input[layer_index + 1]
                # Denoising must be applied.
                if layer.denoise > 1.e-5:
                    self.layer_denoised[layer_index] = GaussianBlur(layer_component_before_denoise, (0, 0),
                        layer.radius / 3., borderType=BORDER_DEFAULT) * layer.denoise + \
                        layer_component_before_denoise * (1. - layer.denoise)
                else:
                    self.layer_denoised[layer_index] = layer_component_before_denoise

        # Build the sharpened image as a weighted sum of the layer components. Start with the
        # maximally blurred layer input (start image of first layer beyond active layers).
        # A weight > 1 increases details at this level, a weight < 1 lowers their visibility.
        new_image = copy(self.layer_input[len(layers)])
        for layer, image_layer_component in zip(layers, self.layer_denoised):
            new_image += image_layer_component * layer.amount

        # In case of "luminance only", insert the new luminance channel into a copy of the original
        # image and change back to BGR.
        if self.color and layers[0].luminance_only:
            new_image_bgr = copy(self.input_image_hsv)
            new_image_bgr[:, :, 2] = new_image
            new_image = cvtColor(new_image_bgr, COLOR_HSV2BGR)

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

        self.tabWidget_postproc_control.currentChanged.connect(self.tab_changed)

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.pushButton_add_layer.clicked.connect(self.add_layer)

        self.comboBox_resolution.addItem('     1 Pixel ')
        self.comboBox_resolution.addItem('   0.5 Pixels')
        self.comboBox_resolution.addItem('  0.25 Pixels')
        self.fgw_slider_value.valueChanged['int'].connect(self.fgw_changed)
        self.comboBox_resolution.currentIndexChanged.connect(self.rgb_resolution_changed)
        self.checkBox_automatic.stateChanged.connect(self.rgb_automatic_changed)
        self.pushButton_red_reset.clicked.connect(self.prreset_clicked)
        self.pushButton_red_up.clicked.connect(self.pru_clicked)
        self.pushButton_red_down.clicked.connect(self.prd_clicked)
        self.pushButton_red_left.clicked.connect(self.prl_clicked)
        self.pushButton_red_right.clicked.connect(self.prr_clicked)
        self.pushButton_blue_reset.clicked.connect(self.pbreset_clicked)
        self.pushButton_blue_up.clicked.connect(self.pbu_clicked)
        self.pushButton_blue_down.clicked.connect(self.pbd_clicked)
        self.pushButton_blue_left.clicked.connect(self.pbl_clicked)
        self.pushButton_blue_right.clicked.connect(self.pbr_clicked)

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
        self.gridLayout.addWidget(self.frame_viewer, 0, 0, 2, 1)

        # Initialize a vertical spacer used to fill the lower part of the sharpening widget scroll
        # area.
        self.spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum,
                                            QtWidgets.QSizePolicy.Expanding)

        # Set the resolution index to an impossible value. It is used to check for changes.
        self.rgb_resolution_index = -1

        # Set the vertical image size (in pixels) to an impossible value. It is used to check for
        # changes later. When the pixel resolution of the displayed image changes, the "fit in view"
        # method of the image viewer is invoked to normalize the appearance of the image.
        self.image_size_y = -1

        # Create the version manager and pass it the "select_version" callback function.
        self.version_manager_widget = VersionManagerWidget(self.configuration, self.select_version)
        self.gridLayout.addWidget(self.version_manager_widget, 1, 1, 1, 1)

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
        self.image_processor.set_shift_display_signal.connect(self.display_shifts)
        self.image_processor.set_status_bar_signal.connect(set_status_bar_callback)

    def fgw_changed(self, value):
        """
        When the widget changes its value, update the corresponding entry for the current version.
        Please note that for some parameters the representations differ.

        The methods following this one do the same for all other configuration parameters.

        :param value: New value sent by widget
        :return: -
        """

        gauss_width = 2 * value - 1
        self.postproc_data_object.versions[self.postproc_data_object.version_selected].rgb_gauss_width = gauss_width
        self.fgw_label_display.setText(str(gauss_width))

    def rgb_resolution_changed(self, value):
        version = self.postproc_data_object.versions[self.postproc_data_object.version_selected]
        version.rgb_resolution_index = value

        # Round the shift values according to the new resolution.
        factor = [1., 2., 4.][value]
        factor_incremented = [1.01, 2.01, 4.01][value]
        version.shift_red = (round(version.shift_red[0] * factor_incremented) / factor ,
                             round(version.shift_red[1] * factor_incremented) / factor)
        version.shift_blue = (round(version.shift_blue[0] * factor_incremented) / factor,
                              round(version.shift_blue[1] * factor_incremented) / factor)
        version.correction_red = (round(version.correction_red[0] * factor_incremented) / factor,
                             round(version.correction_red[1] * factor_incremented) / factor)
        version.correction_blue = (round(version.correction_blue[0] * factor_incremented) / factor,
                              round(version.correction_blue[1] * factor_incremented) / factor)
        self.display_shifts()

    def tab_changed(self, index):
        version = self.postproc_data_object.versions[
            self.postproc_data_object.version_selected]
        if index and not version.rgb_automatic:
            self.rgb_correction_version_init(version)
        elif not index:
            self.finish_rgb_correction_mode()

    def rgb_automatic_changed(self, state):
        rgb_on = state == QtCore.Qt.Checked
        version = self.postproc_data_object.versions[
            self.postproc_data_object.version_selected]
        version.rgb_automatic = rgb_on
        if rgb_on:
            self.rgb_correction_version_reset(version)
        else:
            self.rgb_correction_version_init(version)

    def rgb_correction_version_init(self, version):
        version.rgb_correction_mode = True
        version.correction_red_saved = version.correction_blue_saved = version.correction_red = \
            version.correction_blue = (0., 0.)
        version.images_uncorrected = [None] * 3
        print ("initialize rgb correction for single version")

    def rgb_correction_version_reset(self, version):
        version.rgb_correction_mode = False
        version.correction_red_saved = version.correction_blue_saved = version.correction_red =\
            version.correction_blue = (0., 0.)
        version.images_uncorrected = [None] * 3
        print("reset rgb correction for single version")

    def finish_rgb_correction_mode(self):
        print ("cleaning up rgb corrections")
        for version_index, version in enumerate(self.postproc_data_object.versions):
            if version.correction_red != (0., 0.) or version.correction_blue != (0., 0.):
                print("Applying corrections to version " + str(version_index))
                version.shift_red = (version.shift_red[0] + version.correction_red[0],
                                     version.shift_red[1] + version.correction_red[1])
                version.shift_blue = (version.shift_blue[0] + version.correction_blue[0],
                                      version.shift_blue[1] + version.correction_blue[1])
        self.select_image(self.postproc_data_object.version_selected)

    def prreset_clicked(self):
        version = self.postproc_data_object.versions[self.postproc_data_object.version_selected]
        version.correction_red = (-version.shift_red[0], -version.shift_red[1])
        self.display_shifts()

    def pru_clicked(self):
        version = self.postproc_data_object.versions[self.postproc_data_object.version_selected]
        increment = [1., 0.5, 0.25][version.rgb_resolution_index]
        version.correction_red = (version.correction_red[0] - increment, version.correction_red[1])
        self.display_shifts()

    def prd_clicked(self):
        version = self.postproc_data_object.versions[self.postproc_data_object.version_selected]
        increment = [1., 0.5, 0.25][version.rgb_resolution_index]
        version.correction_red = (version.correction_red[0] + increment, version.correction_red[1])
        self.display_shifts()

    def prl_clicked(self):
        version = self.postproc_data_object.versions[self.postproc_data_object.version_selected]
        increment = [1., 0.5, 0.25][version.rgb_resolution_index]
        version.correction_red = (version.correction_red[0], version.correction_red[1] - increment)
        self.display_shifts()

    def prr_clicked(self):
        version = self.postproc_data_object.versions[self.postproc_data_object.version_selected]
        increment = [1., 0.5, 0.25][version.rgb_resolution_index]
        version.correction_red = (version.correction_red[0], version.correction_red[1] + increment)
        self.display_shifts()

    def pbreset_clicked(self):
        version = self.postproc_data_object.versions[self.postproc_data_object.version_selected]
        version.correction_blue = (-version.shift_blue[0], -version.shift_blue[1])
        self.display_shifts()

    def pbu_clicked(self):
        version = self.postproc_data_object.versions[self.postproc_data_object.version_selected]
        increment = [1., 0.5, 0.25][version.rgb_resolution_index]
        version.correction_blue = (version.correction_blue[0] - increment, version.correction_blue[1])
        self.display_shifts()

    def pbd_clicked(self):
        version = self.postproc_data_object.versions[self.postproc_data_object.version_selected]
        increment = [1., 0.5, 0.25][version.rgb_resolution_index]
        version.correction_blue = (
        version.correction_blue[0] + increment, version.correction_blue[1])
        self.display_shifts()

    def pbl_clicked(self):
        version = self.postproc_data_object.versions[self.postproc_data_object.version_selected]
        increment = [1., 0.5, 0.25][version.rgb_resolution_index]
        version.correction_blue = (version.correction_blue[0], version.correction_blue[1] - increment)
        self.display_shifts()

    def pbr_clicked(self):
        version = self.postproc_data_object.versions[self.postproc_data_object.version_selected]
        increment = [1., 0.5, 0.25][version.rgb_resolution_index]
        version.correction_blue = (
        version.correction_blue[0], version.correction_blue[1] + increment)
        self.display_shifts()

    def display_shifts(self):
        """
        Set the current channel shifts in the RGB alignment GUI tab.

        :return: -
        """

        format_string = ["{0:2.0f}", "{0:4.1f}", "{0:5.2f}"][self.postproc_data_object.versions[
            self.postproc_data_object.version_selected].rgb_resolution_index]

        # Red channel shifts:
        shift_red_base = self.postproc_data_object.versions[
            self.postproc_data_object.version_selected].shift_red
        correction_red = self.postproc_data_object.versions[
            self.postproc_data_object.version_selected].correction_red
        shift_red = (shift_red_base[0] + correction_red[0], shift_red_base[1] + correction_red[1])

        if abs(shift_red[0]) < 0.05:
            self.label_red_down.setText("")
            self.label_red_up.setText("")
        elif shift_red[0] > 0:
            self.label_red_down.setText((format_string.format(shift_red[0])))
            self.label_red_up.setText("")
        else:
            self.label_red_up.setText((format_string.format(-shift_red[0])))
            self.label_red_down.setText("")

        if abs(shift_red[1]) < 0.05:
            self.label_red_left.setText("")
            self.label_red_right.setText("")
        elif shift_red[1] > 0:
            self.label_red_right.setText((format_string.format(shift_red[1])))
            self.label_red_left.setText("")
        else:
            self.label_red_left.setText((format_string.format(-shift_red[1])))
            self.label_red_right.setText("")

        # Blue channel shifts:
        shift_blue_base = self.postproc_data_object.versions[
            self.postproc_data_object.version_selected].shift_blue
        correction_blue = self.postproc_data_object.versions[
            self.postproc_data_object.version_selected].correction_blue
        shift_blue = (shift_blue_base[0] + correction_blue[0], shift_blue_base[1] + correction_blue[1])

        if abs(shift_blue[0]) < 0.05:
            self.label_blue_down.setText("")
            self.label_blue_up.setText("")
        elif shift_blue[0] > 0:
            self.label_blue_down.setText((format_string.format(shift_blue[0])))
            self.label_blue_up.setText("")
        else:
            self.label_blue_up.setText((format_string.format(-shift_blue[0])))
            self.label_blue_down.setText("")

        if abs(shift_blue[1]) < 0.05:
            self.label_blue_left.setText("")
            self.label_blue_right.setText("")
        elif shift_blue[1] > 0:
            self.label_blue_right.setText((format_string.format(shift_blue[1])))
            self.label_blue_left.setText("")
        else:
            self.label_blue_left.setText((format_string.format(-shift_blue[1])))
            self.label_blue_right.setText("")

    def select_version(self, version_index):
        """
        Select a new current version, update the scroll area with all layer widgets, and update the
        parameters of the RGB alignment tab.

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

        # Connect the signal issued by a sharpening layer when the "luminance only" checkbox changes
        # state with all other layers, so they do the same.
        for layer_index, sharpening_layer_widget in enumerate(self.sharpening_layer_widgets):
            toggle_luminance_signal = sharpening_layer_widget.signal_toggle_luminance_other_layers
            for other_layer_index, other_sharpening_layer_widget in enumerate(
                    self.sharpening_layer_widgets):
                if layer_index != other_layer_index:
                    toggle_luminance_signal.connect(
                        other_sharpening_layer_widget.checkBox_luminance_toggled_no_signal)

        # At the end of the scroll area, add a vertical spacer.
        self.verticalLayout.addItem(self.spacerItem)

        # Load the parameters of this version into the RGB alignment tab.
        self.checkBox_automatic.setChecked(version_selected.rgb_automatic)
        self.comboBox_resolution.setCurrentIndex(version_selected.rgb_resolution_index)
        self.fgw_slider_value.setValue(int((version_selected.rgb_gauss_width + 1) / 2))
        self.fgw_label_display.setText(str(version_selected.rgb_gauss_width))

        # Load the current image into the image viewer.
        self.select_image(version_index)

        # Display the current RGB shifts for the selected version.
        self.display_shifts()

    def select_image(self, version_index):
        """
        Load the current image into the image viewer. This method is either invoked by a direct
        call or by a "set_photo_signal" signal.

        :param version_index: Index of the version selected in version list of the central data
                              object.
        :return: -
        """

        image = self.postproc_data_object.versions[version_index].image
        self.frame_viewer.setPhoto(image)

        # Check if the image size has changed by more than 10%. If so, reset the zoom factor of the
        # FrameViewer.
        print ("Select image")
        if abs((image.shape[0] - self.image_size_y)/ image.shape[0]) > 0.1:
            print("Fit in view")
            self.frame_viewer.fitInView()
            self.image_size_y = image.shape[0]

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
    # input_file_name = "D:\SW-Development\Python\PlanetarySystemStacker\Examples\Moon_2018-03-24\Moon_Tile-024_043939_pss.tiff"
    input_file_name = "D:\SW-Development\Python\PlanetarySystemStacker\Examples\Jupiter_Richard\\" \
                      "2020-07-29-2145_3-L-Jupiter_ALTAIRGP224C_pss_p70_b48_rgb-shifted.png"
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
