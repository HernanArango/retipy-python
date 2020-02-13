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

"""tests for tortuosity module"""

from unittest import TestCase
from retipy.retina import Retina
import copy
from numpy.testing import assert_array_equal
from retipy import drusen
import cv2


class TestDrusen(TestCase):
    _resources = 'resources/images/'
    _image_file_name = 'drusen.jpg'
    _image_path = _resources + _image_file_name
    
    

    def setUp(self):
        self.image = cv2.imread(self._image_path)
        

    def test_change_resolution(self):
        image = drusen.change_resolution(self.image)
        height, width, _ = image.shape
        self.assertEqual(700, width)
        self.assertEqual(529, height)

    def test_detection_optical_disc(self):
        little_image = drusen.change_resolution(self.image)
        result = drusen.detect_optical_disc(little_image)
        self.optical_disc = result
        self.assertEqual(result, [142,219])

    """
    def test_detection_roi(self):

        original_image = copy.copy(self.image)
        image = drusen.change_resolution(self.image)

        cols_original, rows_original, _ = original_image.shape
        cols_modified, rows_modified, _ = image.shape
        # Get the original ratio
        Rx = (rows_original/rows_modified)
        Ry = (cols_original/cols_modified)
        
        x,y = drusen.detect_optical_disc(image)
        roi = drusen.detect_roi(self.image, [round(x*Rx), round(y*Ry)])
        b1, g1, r1 = cv2.split(roi)
        b2, g2, r2 = cv2.split(self.roi)
        assert_array_equal(g1,g2)
    """
    def test_total_drusen(self):
        drusen.main(self.image)
        self.assertEqual(drusen.classification_scale["Normal"], 487)
        self.assertEqual(drusen.classification_scale["Medium"], 128)
        self.assertEqual(drusen.classification_scale["Large"], 35)
        

if __name__ == "__main__":
    unittest.main()

