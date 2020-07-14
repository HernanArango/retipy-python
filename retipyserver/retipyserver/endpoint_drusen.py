# Retipy - Retinal Image Processing on Python
# Copyright (C) 2018  Alejandro Valdes
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

"""
Module with endpoints related to server functionality and status.
"""

import flask
from . import app
from . import base_url
from retipy import drusen
from PIL import Image
from retipy.retina import Retina
import numpy as np
import base64
import io
import imutils
import cv2

@app.route(base_url + "drusenclassificationbysize", methods=["POST"])
def post_drusen_classification_by_size():
	
	if flask.request.method == "POST":
		json = flask.request.get_json(silent=True)
		if json is not None:  # pragma: no cover
			image = base64.b64decode(json["image"])
			image = Image.open(io.BytesIO(image))   
			image.save("tmp_drusen.jpg")
			image = cv2.imread("tmp_drusen.jpg")
			drusen_image, classification_scale  = drusen.main(image)
			cv2.imwrite("tmp_drusen.jpg",drusen_image)
			drusen_image = Retina._open_image("tmp_drusen.jpg")
			information = "Total Normal Drusen (<= 63 micron) : "+ str(classification_scale["Normal"])+",Total Medium Drusen (>  63 micron and <= 125 micron) : "+str(classification_scale["Medium"])+",Total Large Drusen  (>  125 micron) : "+str(classification_scale["Large"])+",Normal= Green Color Medium Blue Color Large = Red Color"
			data = {"drusen": Retina.get_base64_image(drusen_image,False), "information": information}
	return flask.jsonify(data)

