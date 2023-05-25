# This script is from the CrossRE project (Bassignana and Plank, 2022)
# See README for details.

from .classifiers import *
from .losses import *


def load_classifier():
	return LinearClassifier, LabelLoss

