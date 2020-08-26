import pickle
from os import walk
import cv2
import numpy as np

def pickle_it(data, path):
	with open(path, 'wb') as f:
		pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
