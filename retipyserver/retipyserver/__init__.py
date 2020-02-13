from flask import Flask
app = Flask(__name__)
base_url = "/retipy/"
from . import endpoint_main
from . import endpoint_tortuosity
from . import endpoint_segmentation
from . import endpoint_landmarks
from . import endpoint_vessel_classification
from . import endpoint_drusen
