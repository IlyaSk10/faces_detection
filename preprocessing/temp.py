import pickle
from os import walk
import cv2
import numpy as np



def load_face_dataset(path):
	face = load_dir(path + 'face\\')
	non_face = load_dir(path + 'non-face\\')

	target = np.hstack((np.ones(len(face)), np.zeros(len(non_face))))
	dataset = np.asarray(face + non_face)

	return dataset, target

def pickle_it(data, path):
	with open(path, 'wb') as f:
		pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
