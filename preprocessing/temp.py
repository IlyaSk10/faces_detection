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


def normalize(dataset):
	std = np.std(dataset.reshape(dataset.shape[0], -1), axis=1)
	mean = np.mean(dataset.reshape(dataset.shape[0], -1), axis=1)
	for i in range(len(std)):
		dataset[i] = (dataset[i] - mean[i]) / std[i]
	return dataset


def pickle_it(data, path):
	with open(path, 'wb') as f:
		pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def unpickle_it(path):
	with open(path, 'rb') as f:
		return pickle.load(f)

