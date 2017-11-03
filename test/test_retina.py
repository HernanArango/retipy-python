# Retipy - Retinal Image Processing on Python
# Copyright (C) 2017  Alejandro Valdes
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""tests for retina module"""

import os
from unittest import TestCase

import cv2
import numpy as np
from numpy.testing import assert_array_equal

from retipy import retina

_resources = 'resources'
_image_file_name = 'im0001.png'
_image_path = _resources + "/images/" + _image_file_name


class TestRetina(TestCase):
    """Test class for Retina class"""

    def setUp(self):
        self.image = retina.Retina(None, _image_path)

    def tearDown(self):
        if os.path.isfile("./out_" + _image_file_name):
            os.unlink("./out_" + _image_file_name)

    def test_constructor_invalid_path(self):
        """Test the retina constructor when the given path is invalid"""
        self.assertRaises(retina.RetinaException, retina.Retina, None, _resources)

    def test_constructor_existing_image(self):
        """Test the constructor with an existing image"""
        image = retina.Retina(None, _image_path)
        none_constructor_image = retina.Retina(image.image, _image_file_name)

        assert_array_equal(image.image, none_constructor_image.image, "created images should be the same")

    def test_segmented(self):
        """Test default value for segmented property"""
        self.assertEqual(
            False, self.image.segmented, "segmented should be false by default")
        self.image.segmented = True
        self.assertEqual(
            True, self.image.segmented, "segmented should be true")

    def test_threshold_image(self):
        self.image.threshold_image()
        _, opencv_output = cv2.threshold(
            cv2.cvtColor(
                cv2.imread(_image_path), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)

        assert_array_equal(self.image.image, opencv_output, "segmented image does not match")

    def test_save_image(self):
        self.image.save_image(".")
        self.assertTrue(os.path.isfile("./out_" + _image_file_name))

    def test_undo(self):
        self.image.detect_edges()
        original_image = retina.Retina(None, _image_path)
        self.assertRaises(
            AssertionError,
            assert_array_equal,
            self.image.image, original_image.image, "images should be different")
        self.image.undo()
        assert_array_equal(self.image.image, original_image.image, "image should be the same")


class TestWindow(TestCase):

    _image_size = 64

    def setUp(self):
        self._retina_image = retina.Retina(
            np.zeros((self._image_size, self._image_size), np.uint8), "window_test")

    def test_constructor(self):
        """test the Window constructor in a positive scenario"""
        retina.Window(self._retina_image, 0, 8, 0, 0)

    def test_create_windows(self):
        # test with an empty image
        windows = retina.create_windows(self._retina_image, 8)
        self.assertTrue(not windows, "windows should be empty")

        # test with a full data image
        self._retina_image.image[:,:] = 1
        windows = retina.create_windows(self._retina_image, 8)
        self.assertEqual(len(windows), self._image_size, "expected 64 windows")

        # test with an image half filled with data
        self._retina_image.image[:, 0:int(self._image_size/2)] = 0
        windows = retina.create_windows(self._retina_image, 8)
        self.assertEqual(len(windows), self._image_size/2, "expected 32 windows")

    def test_create_windows_combined(self):
        windows = retina.create_windows(self._retina_image, 8, "combined", -1)

        # combined should create (width/(dimension/2) - 1) * (height/(dimension/2) -1)
        # here is (64/4 -1) * (64/4 -1) = 225
        self.assertEqual(len(windows), 225, "there should be 225 windows created")

        windows = retina.create_windows(self._retina_image, 8, "combined")
        self.assertFalse(windows, "no window should be created")

    def test_vessel_extractor(self):
        self._retina_image.image[10, 10:20] = 1
        vessels = retina.detect_vessel_border(self._retina_image)

        self.assertEqual(len(vessels), 1, "only one vessel should've been extracted")
        self.assertEqual(len(vessels[0]), 10, "vessel should have 10 pixels")
