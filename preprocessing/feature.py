import pickle
from os import walk
import cv2
import numpy as np


def load_dir(path):
	data = []
	for (dirpath, dirnames, filenames) in walk(path):
		for filename in filenames:
			data.append(cv2.imread(dirpath + filename, -1))
		break

	return data


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


class Features:
	def __init__(self, data):
		self.data = data
		self.y = len(data[0, 0])
		self.x = len(data[0, :, 0])
		self.integral_data = np.copy(data)
		self.features_data = []
		super().__init__()

	def integral_image_create(self):
		for image in self.integral_data.astype(int):

			for y in range(len(image)):
				image[:, y] = np.cumsum(image[:, y])

			for x in range(len(image[0])):
				image[x] = np.cumsum(image[x])

	# TODO: Refactor this mess with a copied code
	# So much to do and so little time for it
	class FirstFeature:
		def __init__(self, integral_data):
			self.integral_data = integral_data
			self.features = []
			self.y = len(integral_data[0, 0])
			self.x = len(integral_data[0, :, 0])

		@staticmethod
		def window_init(x_size, y_size):
			a = (-1, -1)
			b = (-1, x_size - 1)
			c = (y_size - 1, -1)
			d = (y_size - 1, x_size - 1)
			e = (-1, (x_size * 2) - 1)
			f = (y_size - 1, (x_size * 2) - 1)
			return a, b, c, d, e, f

		@staticmethod
		def window_move(a, b, c, d, e, f, stride, dim):
			if dim == 1:
				move = np.array([0, stride], dtype=int)
			else:
				move = np.array([stride, 0], dtype=int)
			return a + move, b + move, c + move, d + move, e + move, f + move, move

		def generate(self):
			for image in range(len(self.integral_data)):
				features = []
				for y_size in range(self.y):
					for x_size in range(self.x):

						# Size cannot be zero
						if (x_size == 0) or (y_size == 0):
							continue

						# If kernel > size of image in any direction, then we exit
						if (2 * y_size >= self.integral_data[0].shape[0]) or (
								2 * x_size >= self.integral_data[0].shape[1]):
							break

						a, b, c, d, e, f = self.window_init(x_size, y_size)

						for y in range(self.y):
							for x in range(self.x):

								if (f[0] >= self.y) or (f[1] >= self.x):
									break

								white = self.integral_data[image, d[0], d[1]]
								black = self.integral_data[image, f[0], f[1]] - self.integral_data[image, d[0], d[1]]

								if a[0] == -1:  # If in top
									if c[1] == -1:  # If in corner
										features.append(white - black)
									else:
										white -= self.integral_data[image, c[0], c[1]]
										features.append(white - black)
								else:
									white -= self.integral_data[image, b[0], b[1]]
									black -= (self.integral_data[image, e[0], e[1]] +
									          self.integral_data[image, b[0], b[1]])
									if a[1] == -1:  # If left side but not top
										features.append(white - black)
									else:
										white -= (self.integral_data[image, c[0], c[1]] +
										          self.integral_data[image, a[0], a[1]])
										features.append(white - black)

								a, b, c, d, e, f, move = self.window_move(a, b, c, d, e, f, 1, dim=1)

							a, b, c, d, e, f = self.window_init(x_size, y_size)
							a, b, c, d, e, f, move = self.window_move(a, b, c, d, e, f, stride=y, dim=0)
				self.features.append(features)
			return np.array(self.features)

	class SecondFeature(FirstFeature):
		def __init__(self, integral_data):
			super().__init__(integral_data)

		def window_init_ext(self, x_size, y_size):
			a, b, c, d, e, f = self.window_init(x_size, y_size)
			g = (-1, (x_size * 3) - 1)
			h = (y_size - 1, (x_size * 3) - 1)
			return a, b, c, d, e, f, g, h

		def window_move_ext(self, a, b, c, d, e, f, g, h, stride, dim):
			a, b, c, d, e, f, move = self.window_move(a, b, c, d, e, f, stride=stride, dim=dim)
			return a, b, c, d, e, f, g + move, h + move

		def generate(self):
			for image in range(len(self.integral_data)):
				features = []
				for y_size in range(self.y):
					for x_size in range(self.x):

						# Size cannot be zero
						if (x_size == 0) or (y_size == 0):
							continue

						# If kernel > size of image in any direction, then we exit
						if (2 * y_size >= self.integral_data[0].shape[0]) or (
								2 * x_size >= self.integral_data[0].shape[1]):
							break

						a, b, c, d, e, f, g, h = self.window_init_ext(x_size, y_size)

						for y in range(self.y):
							for x in range(self.x):

								if (h[0] >= self.y) or (h[1] >= self.x):
									break

								white1 = self.integral_data[image, d[0], d[1]]
								black = self.integral_data[image, f[0], f[1]] - self.integral_data[image, d[0], d[1]]
								white2 = self.integral_data[image, h[0], h[1]]

								if a[0] == -1:  # If in top
									if c[1] == -1:  # If in corner
										features.append(white1 + white2 - black)
									else:
										white1 -= self.integral_data[image, c[0], c[1]]
										white2 -= self.integral_data[image, f[0], f[1]]
										features.append(white1 + white2 - black)
								else:
									white1 -= self.integral_data[image, b[0], b[1]]

									white2 -= (self.integral_data[image, g[0], g[1]] +
									           self.integral_data[image, e[0], e[1]])

									black -= (self.integral_data[image, e[0], e[1]] +
									          self.integral_data[image, b[0], b[1]])
									if a[1] == -1:  # If left side but not top
										features.append(white1 + white2 - black)
									else:
										white1 -= (self.integral_data[image, c[0], c[1]] +
										           self.integral_data[image, a[0], a[1]])
										features.append(white1 + white2 - black)

								a, b, c, d, e, f, g, h = self.window_move_ext(a, b, c, d, e, f, g, h, 1, dim=1)

							a, b, c, d, e, f, g, h = self.window_init_ext(x_size, y_size)
							a, b, c, d, e, f, g, h = self.window_move_ext(a, b, c, d, e, f, g, h, stride=y, dim=0)
				self.features.append(features)
			return np.array(self.features)

	class ThirdFeature(FirstFeature):
		def __init__(self, integer_data):
			super().__init__(integer_data)

		def window_init_alt(self, x_size, y_size):
			a, b, c, d, e, f = self.window_init(x_size, y_size)
			e = ((y_size * 2) - 1, -1)
			f = ((y_size * 2) - 1, x_size - 1,)
			return a, b, c, d, e, f

		def window_move_alt(self, a, b, c, d, e, f, stride, dim):
			a, b, c, d, e, f, move = self.window_move(a, b, c, d, e, f, stride=stride, dim=dim)
			return a, b, c, d, e, f, move

		def generate(self):
			for image in range(len(self.integral_data)):
				features = []
				for y_size in range(self.y):
					for x_size in range(self.x):

						# Size cannot be zero
						if (x_size == 0) or (y_size == 0):
							continue

						# If kernel > size of image in any direction, then we exit
						if (2 * y_size >= self.integral_data[0].shape[0]) or (
								2 * x_size >= self.integral_data[0].shape[1]):
							break

						a, b, c, d, e, f = self.window_init_alt(x_size, y_size)

						for y in range(self.y):
							for x in range(self.x):

								if (f[0] >= self.y) or (f[1] >= self.x):
									break

								white = self.integral_data[image, d[0], d[1]]
								black = self.integral_data[image, f[0], f[1]] - \
								        self.integral_data[image, d[0], d[1]]

								if a[0] == -1:  # If in top
									if c[1] == -1:  # If in corner
										features.append(white - black)
									else:
										white -= self.integral_data[image, c[0], c[1]]
										black -= (self.integral_data[image, e[0], e[1]] +
										          self.integral_data[image, c[0], c[1]])
										features.append(white - black)
								else:
									white -= self.integral_data[image, b[0], b[1]]
									if a[1] == -1:  # If left side but not top
										features.append(white - black)
									else:
										white -= (self.integral_data[image, c[0], c[1]] +
										          self.integral_data[image, a[0], a[1]])
										black -= (self.integral_data[image, e[0], e[1]] +
										          self.integral_data[image, c[0], c[1]])
										features.append(white - black)

								a, b, c, d, e, f, move = self.window_move_alt(a, b, c, d, e, f, 1, dim=1)

							a, b, c, d, e, f = self.window_init_alt(x_size, y_size)
							a, b, c, d, e, f, move = self.window_move_alt(a, b, c, d, e, f, stride=y, dim=0)
				self.features.append(features)
			return np.array(self.features)

	class FourthFeature(ThirdFeature):
		def __init__(self, integer_data):
			super().__init__(integer_data)

		def window_init_alt_ext(self, x_size, y_size):
			a, b, c, d, e, f = self.window_init_alt(x_size, y_size)
			g = ((y_size * 3) - 1, -1)
			h = ((y_size * 3) - 1, x_size-1)
			return a, b, c, d, e, f, g, h

		def window_move_alt_ext(self, a, b, c, d, e, f, g, h, stride, dim):
			a, b, c, d, e, f, move = self.window_move_alt(a, b, c, d, e, f, stride=stride, dim=dim)
			return a, b, c, d, e, f, g + move, h + move

		def generate(self):
			for image in range(len(self.integral_data)):
				features = []
				for y_size in range(self.y):
					for x_size in range(self.x):

						# Size cannot be zero
						if (x_size == 0) or (y_size == 0):
							continue

						# If kernel > size of image in any direction, then we exit
						if (2 * y_size >= self.integral_data[0].shape[0]) or (
								2 * x_size >= self.integral_data[0].shape[1]):
							break

						a, b, c, d, e, f, g, h = self.window_init_alt_ext(x_size, y_size)

						for y in range(self.y):
							for x in range(self.x):

								if (h[0] >= self.y) or (h[1] >= self.x):
									break

								white1 = self.integral_data[image, d[0], d[1]]
								black = self.integral_data[image, f[0], f[1]] - \
								        self.integral_data[image, d[0], d[1]]
								white2 = self.integral_data[image,h[0], h[1]] - \
								         self.integral_data[image, f[0], f[1]]

								if a[0] == -1:  # If in top
									if c[1] == -1:  # If in corner
										features.append(white1 + white2 - black)
									else:
										white1 -= self.integral_data[image, c[0], c[1]]
										black -= (self.integral_data[image, e[0], e[1]] +
										          self.integral_data[image, c[0], c[1]])
										white2 -= (self.integral_data[image, g[0], g[1]] +
										          self.integral_data[image, e[0], e[1]])
										features.append(white1 + white2 - black)
								else:
									white1 -= self.integral_data[image, b[0], b[1]]
									if a[1] == -1:  # If left side but not top
										features.append(white1 - black)
									else:
										white1 -= (self.integral_data[image, c[0], c[1]] +
										          self.integral_data[image, a[0], a[1]])
										black -= (self.integral_data[image, e[0], e[1]] +
										          self.integral_data[image, c[0], c[1]])
										white2 -= (self.integral_data[image, g[0], g[1]] +
										           self.integral_data[image, e[0], e[1]])
										features.append(white1 + white2 - black)

								a, b, c, d, e, f, g, h = self.window_move_alt_ext(a, b, c, d, e, f, g, h, 1, dim=1)

							a, b, c, d, e, f, g, h = self.window_init_alt_ext(x_size, y_size)
							a, b, c, d, e, f, g, h = self.window_move_alt_ext(a, b, c, d, e, f, g, h, stride=y, dim=0)
				self.features.append(features)
			return np.array(self.features)

	class FifthFeature(FirstFeature):
		def __init__(self, integer_data):
			super().__init__(integer_data)

		def window_init_ext(self, x_size, y_size):
			a, b, c, d, e, f = self.window_init(x_size, y_size)
			g = ((y_size * 2) - 1, -1)
			h = ((y_size * 2) - 1, x_size - 1)
			i = ((y_size * 2) - 1, (x_size * 2) - 1)
			return a, b, c, d, e, f, g, h, i

		def window_move_ext(self, a, b, c, d, e, f, g, h, i, stride, dim):
			a, b, c, d, e, f, move = self.window_move(a, b, c, d, e, f, stride=stride, dim=dim)
			return a, b, c, d, e, f, g + move, h + move, i + move

		def generate(self):
			for image in range(len(self.integral_data)):
				features = []
				for y_size in range(self.y):
					for x_size in range(self.x):

						# Size cannot be zero
						if (x_size == 0) or (y_size == 0):
							continue

						# If kernel > size of image in any direction, then we exit
						if (2 * y_size >= self.integral_data[0].shape[0]) or (
								2 * x_size >= self.integral_data[0].shape[1]):
							break

						a, b, c, d, e, f, g, h, i = self.window_init_ext(x_size, y_size)

						for y in range(self.y):
							for x in range(self.x):

								if (i[0] >= self.y) or (i[1] >= self.x):
									break

								white1 = self.integral_data[image, d[0], d[1]]
								black1 = self.integral_data[image, f[0], f[1]] - \
								        self.integral_data[image, d[0], d[1]]

								white2 = self.integral_data[image, i[0], i[1]] - \
								        self.integral_data[image, f[0], f[1]] - \
										 self.integral_data[image, h[0], h[1]] + \
										 self.integral_data[image, d[0], d[1]]
								black2 = self.integral_data[image, h[0], h[1]] - \
								        self.integral_data[image, d[0], d[1]]

								if a[0] == -1:  # If in top
									if c[1] == -1:  # If in corner
										features.append(white1 + white2 - black1 - black2)
									else:
										white1 -= self.integral_data[image, c[0], c[1]]
										black2 -= (self.integral_data[image, g[0], g[1]] +
										           self.integral_data[image, c[0], c[1]])
										features.append(white1 + white2 - black1 - black2)
								else:
									white1 -= self.integral_data[image, b[0], b[1]]
									black1 -= (self.integral_data[image, e[0], e[1]] +
									           self.integral_data[image, b[0], b[1]])
									if a[1] == -1:  # If left side but not top
										features.append(white1 + white2 - black1 - black2)
									else:
										white1 -= (self.integral_data[image, c[0], c[1]] +
										           self.integral_data[image, a[0], a[1]])
										black2 -= (self.integral_data[image, g[0], g[1]] +
										          self.integral_data[image, c[0], c[1]])
										features.append(white1 + white2 - black1 - black2)

								a, b, c, d, e, f, g, h, i = self.window_move_ext(a, b, c, d, e, f, g, h, i, 1, dim=1)

							a, b, c, d, e, f, g, h, i = self.window_init_ext(x_size, y_size)
							a, b, c, d, e, f, g, h, i = self.window_move_ext(a, b, c, d, e, f, g, h, i, stride=y, dim=0)
				self.features.append(features)
			return np.array(self.features)

	def temp(self, ftr, text, number):
		print(str(number)+' feature..')
		self.features_data = ftr.generate()
		pickle_it(self.features_data, (text + '_'+str(number)))
		print('Done. \n')

	def generate(self, text):
		self.integral_image_create()

		print('Beginning..')

		ftr = self.FirstFeature(self.integral_data)
		self.temp(ftr, text, 1)

		ftr = self.SecondFeature(self.integral_data)
		self.temp(ftr, text, 2)

		ftr = self.ThirdFeature(self.integral_data)
		self.temp(ftr, text, 3)

		ftr = self.FourthFeature(self.integral_data)
		self.temp(ftr, text, 4)

		ftr = self.FifthFeature(self.integral_data)
		self.temp(ftr, text, 5)


# x_train, t_train = load_face_dataset('faces\\train\\')
x_test, t_test = load_face_dataset('faces\\test\\')

# x_train = normalize(x_train.astype(float))
#x_test = normalize(x_test.astype(float))

# haar_like_features_generator = Features(x_train)
# x_train = haar_like_features_generator.generate('train')

i = 0
j = 4088
for k in range(5):
	temp = x_test[i:j]
	haar_like_features_generator = Features(temp)
	haar_like_features_generator.generate('test_'+str(k))
	i += 4089
	j += 4089

