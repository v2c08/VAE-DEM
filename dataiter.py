from PIL import ImageEnhance
from PIL import Image as pil_image
import numpy as np
import re
from six.moves import range
import os
import threading
import warnings
import multiprocessing.pool
import json
from keras.preprocessing.text import Tokenizer, one_hot
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
import itertools

class Iterator(object):

	def __init__(self,
				 directory,
				 mode,
				 n,
				 batch_size,
				 target_size=(64, 64),
				 color_mode='rgb',
				 classes=None,
				 class_mode=None,
				 seed=None,
				 data_format='channels_last',
				 interpolation='nearest',
				 dtype='float32',
				 latent_data=None):

		self.mode = mode
		self.n = n
		self.batch_size = batch_size
		self.target_size = target_size
		self.color_mode = color_mode
		self.seed = seed
		self.batch_index = 0
		self.total_batches_seen = 0
		self.lock = threading.Lock()
		self.image_index_array = None
		self.json_index = None
		self.image_index_generator = self._flow_image_index()
		self.json_index_generator = self._flow_json_index()
		self.directory = directory
		self.classes = classes
		self.interpolation=interpolation
		self.data_format=data_format
		self.dtype = dtype

		shapes = ['Cube', 'Cone', 'Cylinder', 'Icosphere', 'Torus']
		colours = ['r', 'g', 'b', 'w']
		self.num_shapes = len(shapes)

		labels = []
		for c, r in enumerate(itertools.product(colours, shapes)):
			labels.append(r)

		self.mlb = MultiLabelBinarizer()
		self.mlb.fit(labels)
		self.nClasses = len(self.mlb.classes_)

		self.mlb.transform(self.mlb.classes_)
		follow_links = True

		if self.color_mode == 'rgba':
			self.image_shape = self.target_size + (4,)
		elif self.color_mode == 'rgb':
			self.image_shape = self.target_size + (3,)

		# First, count the number of samples and classes.
		self.samples = 0

		if not classes:
			classes = []
			for subdir in sorted(os.listdir(directory)):
				if os.path.isdir(os.path.join(directory, subdir)):
					classes.append(subdir)
		self.num_classes = len(classes)
		self.class_indices = dict(zip(classes, range(len(classes))))

		pool = multiprocessing.pool.ThreadPool()

		# Second, build an index of the images
		# in the different class subfolders.
		image_results = []
		json_results  = []
		self.image_filenames = []
		self.json_filenames  = []
		i = 0
		for dirpath in (os.path.join(directory, subdir) for subdir in classes):
			image_results.append(
				pool.apply_async(_list_valid_image_filenames_in_directory,
								 (dirpath, ['png'], self.class_indices, follow_links)))
			json_results.append(
			pool.apply_async(_list_valid_image_filenames_in_directory,
								 (dirpath, ['json'], self.class_indices, follow_links)))

		classes_list = []

		for imres,jres in zip(image_results,json_results):
			classes, image_filenames = imres.get()
			classes_list.append(classes)
			self.image_filenames += image_filenames
			_, json_filenames = jres.get()
			self.json_filenames+=json_filenames

		self.samples = len(self.image_filenames)
		self.trials  = len(self.json_filenames)

		self.classes = np.zeros((self.samples,), dtype='int32')
		for classes in classes_list:
			self.classes[i:i + len(classes)] = classes
			i += len(classes)

		if not latent_data is None:
			self.latent_generator = latent_data

		print('Found {} images, {} trials and {} classes.'.format(self.samples, self.trials, self.num_classes))
		pool.close()
		pool.join()

	def _set_image_index_array(self):
		self.image_index_array = np.arange(self.n)

	def _set_json_index(self):
		self.json_index = np.arange(self.n // self.batch_size)

	def __getitem__(self, idx):
		if idx >= len(self):
			raise ValueError('Asked to retrieve element {idx}, '
							 'but the Sequence '
							 'has length {length}'.format(idx=idx,
														  length=len(self)))
		if self.seed is not None:
			np.random.seed(self.seed + self.total_batches_seen)
		self.total_batches_seen += 1
		if self.image_index_array is None:
			self._set_image_index_array()
			self._set_json_index()
		image_index_array = self.image_index_array[self.batch_size * idx:
									   self.batch_size * (idx + 1)]
		return self._get_batches_of_images(image_index_array)

	def __len__(self):
		return (self.n + self.batch_size - 1) // self.batch_size  # round up

	def on_epoch_end(self):
		self._set_image_index_array()

	def reset(self):
		self.batch_index = 0

	def _flow_image_index(self):
		# Ensure self.batch_index is 0.
		self.reset()
		while 1:
			if self.seed is not None:
				np.random.seed(self.seed + self.total_batches_seen)
			if self.batch_index == 0:
				self._set_image_index_array()

			current_index = (self.batch_index * self.batch_size) % self.n
			if self.n > current_index + self.batch_size:
				self.batch_index += 1
			else:
				self.batch_index = 0
			self.total_batches_seen += 1
			yield self.image_index_array[current_index:
								   current_index + self.batch_size]

	def _flow_json_index(self):
		while 1:
			if self.seed is not None:
				np.random.seed(self.seed + self.total_batches_seen)
			if self.batch_index == 0:
				self._set_json_index()
			yield self.json_index[self.batch_index]


	def __iter__(self):
		# Needed if we want to do something like:
		# for x, y in data_gen.flow(...):
		return self

	def __next__(self, *args, **kwargs):
		return self.next(*args, **kwargs)

	def _get_batches_of_images(self, image_index_array):

		batch_x = np.zeros((len(image_index_array),) + self.image_shape, dtype=self.dtype)
		# build batch of image data
		for i, j in enumerate(image_index_array):
			fname = self.image_filenames[j]
			img = self.load_img(os.path.join(self.directory, fname),
						   color_mode=self.color_mode,
						   target_size=self.target_size,
						   interpolation=self.interpolation)
			x = img_to_array(img, data_format=self.data_format)

			# Pillow images should be closed after `load_img`,
			# but not PIL images.
			if hasattr(img, 'close'):
				img.close()

			batch_x[i,:,:,:] = x

		#batch_y = np.roll(batch_x,1,axis=0)

		return batch_x, batch_x

	def _get_batches_of_points(self, json_index, image_index_array):

		json_fname = self.json_filenames[json_index]

		with open(os.path.join(self.directory,json_fname), encoding='utf-8') as json_data:
			data = json.load(json_data)

		#points    = np.zeros((len(image_index_array),data[0]['data']['num_verts'],3), dtype=self.dtype)
		#rotations = np.zeros((len(image_index_array), 3), dtype=self.dtype)
		#rotvels   = np.zeros((len(image_index_array), 3), dtype=self.dtype)
		shapes    = np.zeros((len(image_index_array), 8), dtype=self.dtype)

		for i, obj_data in enumerate(data):

			obj  = obj_data['data']
			points[:, :]   = np.array(obj['point_cloud'])
			rotations[:, :]   = np.array(obj['local_rotation'])
			rotvels[:, :] =  np.array(obj['local_rotvel'])
			shapes[:, :] =  self.tokenizer.texts_to_matrix([data[0]['shape']])

		#points_y = np.roll(points,1,axis=0)
		#rotations_y = np.roll(rotations,1,axis=0)
		#rotvels_y = np.roll(rotations,1,axis=0)

		rotations = np.divide(rotations, 360)

		#return [points, points_y],  [rotations, rotations_y], [rotvels,rotvels_y], [data[0]['shape'],data[0]['shape']]
		#return [rotations, rotations_y], [shapes, shapes]
		return [rotations, rotations], [shapes, shapes]


	def _get_batches_of_labels(self, json_index, image_index_array):

		json_fname = self.json_filenames[json_index]
		n_images = len(image_index_array)

		with open(os.path.join(self.directory,json_fname), encoding='utf-8') as json_data:
			data = json.load(json_data)

		shape  = data[0]['shape']
		colour = data[0]['colour']

		label = self.mlb.transform([[shape, colour]])
		labels = np.repeat(label,n_images, axis=0)
		return labels


	def load_img(self,path, grayscale=False, color_mode='rgb', target_size=None,
				 interpolation='nearest'):

		img = pil_image.open(path)


		#if color_mode == 'grayscale':
		#    if img.mode != 'L':
		#        img = img.convert('L')
		#elif color_mode == 'rgba':
		#    if img.mode != 'RGBA':
		#        img = img.convert('RGBA')
		if color_mode == 'rgb':
			if img.mode != 'RGB':

				img = np.array(img)

				img = 1./255 * img.astype(self.dtype)

		#img = img[:,:,:3]
		#else:
		#    raise ValueError('color_mode must be "grayscale", "rgb", or "rgba"')
		#if target_size is not None:
		#    width_height_tuple = (target_size[1], target_size[0])
		#    if img.size != width_height_tuple:
		#        img = img.resize(width_height_tuple, pil_image.NEAREST)
		return img

	def next(self):

		if self.image_index_array is None:
			self._set_image_index_array()
			self._set_json_index()

		with self.lock:
			image_index_array = next(self.image_index_generator)
			json_index = next(self.json_index_generator)
		# The transformation of images is not under thread lock
		# so it can be done in parallel

		images = self._get_batches_of_images(image_index_array)
		#points, rots, rotvels, shapes = self._get_batches_of_points(json_index, image_index_array)

		labels = self._get_batches_of_labels(json_index, image_index_array)
		#shape, colour = self.get_labels()

		# mst be tuple of format: ([in1, in2, in3], [out1, outn])
		#return images, points, rots, rotvels, shapes
		if self.mode == 'classifier':
			return(images[0], labels)
		elif self.mode == 'cvae':
			return ([images[0], rots[0], shapes[0]], images[1])
		elif self.mode == 'bvae':
			return(images[0], images[1])
		else:
			print('invalid_mode')
			0/0


def _list_valid_image_filenames_in_directory(directory, white_list_formats, class_indices, follow_links, df=False):

	dirname = os.path.basename(directory)

	valid_files = _iter_valid_files(directory, white_list_formats, follow_links)
	if df:
		image_filenames = []
		for root, fname in valid_files:
			image_filenames.append(os.path.basename(fname))
		return image_filenames
	classes = []
	image_filenames = []
	for root, fname in valid_files:
		classes.append(class_indices[dirname])
		absolute_path = os.path.join(root, fname)
		relative_path = os.path.join(
			dirname, os.path.relpath(absolute_path, directory))
		image_filenames.append(relative_path)

	return classes, image_filenames

def _iter_valid_files(directory, white_list_formats, follow_links):
	def _recursive_list(subpath):
		return sorted(os.walk(subpath, followlinks=follow_links),
					  key=lambda x: x[0])

	for root, _, files in _recursive_list(directory):
		for fname in sorted(files):
			for extension in white_list_formats:
				if fname.lower().endswith('.' + extension):
					yield root, fname

def img_to_array(img, data_format='channels_last', dtype='float32'):
	"""Converts a PIL Image instance to a Numpy array.
	# Arguments
		img: PIL Image instance.
		data_format: Image data format,
			either "channels_first" or "channels_last".
		dtype: Dtype to use for the returned array.
	# Returns
		A 3D Numpy array.
	# Raises
		ValueError: if invalid `img` or `data_format` is passed.
	"""
	if data_format not in {'channels_first', 'channels_last'}:
		raise ValueError('Unknown data_format: %s' % data_format)
	# Numpy array x has format (height, width, channel)
	# or (channel, height, width)
	# but original PIL image has format (width, height, channel)
	x = np.asarray(img, dtype=dtype)

	if len(x.shape) == 3:
		if data_format == 'channels_first':
			x = x.transpose(2, 0, 1)
	elif len(x.shape) == 2:
		if data_format == 'channels_first':
			x = x.reshape((1, x.shape[0], x.shape[1]))
		else:
			x = x.reshape((x.shape[0], x.shape[1], 1))
	else:
		raise ValueError('Unsupported image shape: %s' % (x.shape,))
	return x
