import numpy as np
from os import path
import keras.backend as K
from keras.models import Model
from model_util import ConvBnLRelu, SampleLayer
from keras.layers import Input, MaxPool2D, Conv2D, GlobalAveragePooling2D, Reshape, UpSampling2D, Flatten, Dense, BatchNormalization, Activation, Dropout


K.set_image_data_format('channels_last')

class Models():

	def __init__(self,P):
		# Parameter dictionary
		self.P = P

	def _filedef(self, mode):
		weights = path.join('weights', '{}.hdf5'.format(mode))  # where weights will be saved
		json    = path.join('weights', '{}.json'.format(mode))
		loss    = path.join('weights', '{}_loss.txt'.format(mode))
		return weights, json, loss

	def build_models(self):
		import model_util

		# ------- Encoder ------- #

		# Image convolution
		print(type(self.P['image_shape']))
		image_inputs  = Input(shape=self.P['image_shape'], name='image_inputs')

		net = ConvBnLRelu(32, kernelSize=3)(image_inputs) # 1
		net = MaxPool2D((2, 2), strides=(2, 2))(net)

		net = ConvBnLRelu(64, kernelSize=3)(net) # 2
		net = MaxPool2D((2, 2), strides=(2, 2))(net)

		net = ConvBnLRelu(128, kernelSize=3)(net) # 3
		net = ConvBnLRelu(64, kernelSize=1)(net) # 4
		net = ConvBnLRelu(128, kernelSize=3)(net) # 5
		net = MaxPool2D((2, 2), strides=(2, 2))(net)

		net = ConvBnLRelu(256, kernelSize=3)(net) # 6
		net = ConvBnLRelu(128, kernelSize=1)(net) # 7
		net = ConvBnLRelu(256, kernelSize=3)(net) # 8
		net = MaxPool2D((2, 2), strides=(2, 2))(net)

		net = ConvBnLRelu(512, kernelSize=3)(net) # 9
		net = ConvBnLRelu(256, kernelSize=1)(net) # 10
		net = ConvBnLRelu(512, kernelSize=3)(net) # 11
		net = ConvBnLRelu(256, kernelSize=1)(net) # 12
		net = ConvBnLRelu(512, kernelSize=3)(net) # 13
		net = MaxPool2D((2, 2), strides=(2, 2))(net)

		net = ConvBnLRelu(1024, kernelSize=3)(net) # 14
		net = ConvBnLRelu(512, kernelSize=1)(net) # 15
		net = ConvBnLRelu(1024, kernelSize=3)(net) # 16
		net = ConvBnLRelu(512, kernelSize=1)(net) # 17
		net = ConvBnLRelu(1024, kernelSize=3)(net) # 18

		mean = Conv2D(filters=self.P['latent_dim'],
					  kernel_size=(1, 1),
					  padding='same')(net)

		mean = GlobalAveragePooling2D()(mean)

		stddev = Conv2D(filters=self.P['latent_dim'],
						kernel_size=(1, 1),
						padding='same')(net)

		stddev = GlobalAveragePooling2D()(stddev)

		z = SampleLayer()([mean, stddev])

		encoder = Model(image_inputs, z, name='encoder')

		#------- Decoder -------#

		d_in = Input(shape=(self.P['latent_dim'],))
		net = Reshape((1, 1, self.P['latent_dim']))(d_in)
		# darknet downscales input by a factor of 32, so we upsample to the second to last output shape:
		net = UpSampling2D((self.P['im_height']//32, self.P['im_width']//32))(net)

		net = ConvBnLRelu(1024, kernelSize=3)(net)
		net = ConvBnLRelu(512, kernelSize=1)(net)
		net = ConvBnLRelu(1024, kernelSize=3)(net)
		net = ConvBnLRelu(512, kernelSize=1)(net)
		net = ConvBnLRelu(1024, kernelSize=3)(net)

		net = UpSampling2D((2, 2))(net)
		net = ConvBnLRelu(512, kernelSize=3)(net)
		net = ConvBnLRelu(256, kernelSize=1)(net)
		net = ConvBnLRelu(512, kernelSize=3)(net)
		net = ConvBnLRelu(256, kernelSize=1)(net)
		net = ConvBnLRelu(512, kernelSize=3)(net)

		net = UpSampling2D((2, 2))(net)
		net = ConvBnLRelu(256, kernelSize=3)(net)
		net = ConvBnLRelu(128, kernelSize=1)(net)
		net = ConvBnLRelu(256, kernelSize=3)(net)

		net = UpSampling2D((2, 2))(net)
		net = ConvBnLRelu(128, kernelSize=3)(net)
		net = ConvBnLRelu(64, kernelSize=1)(net)
		net = ConvBnLRelu(128, kernelSize=3)(net)

		net = UpSampling2D((2, 2))(net)
		net = ConvBnLRelu(64, kernelSize=3)(net)

		net = UpSampling2D((2, 2))(net)
		net = ConvBnLRelu(64, kernelSize=1)(net)
		net = ConvBnLRelu(32, kernelSize=3)(net)
		decoded = Conv2D(filters=self.P['n_channels'],
				  kernel_size=(1, 1), padding='same')(net)

		decoder = Model(d_in, decoded, name='decoder')
		bvae = Model(encoder.inputs, decoder(encoder.outputs))

		def xent_loss(y_true, y_pred):
			recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
			return recon

		def KL_loss(y_true, y_pred):
			return(0.5 * K.sum(K.exp(stddev) + K.square(mean) - 1. - stddev, axis=1))

		def recon_loss(y_true, y_pred):
			return(K.sum(K.binary_crossentropy(y_pred, y_true), axis=1))

		bvae.compile(optimizer='adam', loss='mean_absolute_error', metrics=[KL_loss, recon_loss])

		# ------- Classifier ------- #
		#latent_inputs  = Input(shape=(self.P['latent_dim'],), name='classifier_inputs')
		x = encoder(encoder.inputs)
		x = Dense(128)(x)
		x = Activation('relu')(x)
		x = BatchNormalization()(x)
		x = Dropout(0.5)(x)
		x = Dense(self.P['n_classes'])(x)
		classification_output = Activation('sigmoid')(x)
		classifier = Model(encoder.inputs, classification_output)
		classifier.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])

		self.classifier = classifier
		self.encoder = encoder
		self.decoder = decoder
		self.bvae = bvae

		return encoder, decoder, bvae, classifier

	def train_bvae(self):

		weights_file, json_file, loss_file = self._filedef('bvae')
		bvae_history = self.bvae.fit_generator(self.datagen('train', 'bvae'),
											   epochs=self.P['epochs'],
											   steps_per_epoch=self.P['train_batches'],
											   validation_data=self.datagen('train', 'bvae'),
											   validation_steps=60,
											   callbacks=self.callbacks(weights_file),
											   workers=1)

		json_string = self.bvae.to_json()
		with open(json_file, 'w') as f:
			f.write(json_string)
		self.bvae.save(weights_file)

		json_string = self.encoder.to_json()
		with open('weights/encoder.json', 'w') as f:
			f.write(json_string)
		self.bvae.save('weights/encoder.hdf5')

		loss_history = np.array(bvae_history.history["loss"])
		np.savetxt(loss_file, loss_history, delimiter=",")
		return

	def train_classifier(self):

		weights_file, json_file, loss_file = self._filedef('classifier')
		classifier_history = self.classifier.fit_generator(self.datagen('train', 'classifier'),
										 epochs=self.P['epochs'],
										 steps_per_epoch=self.P['train_batches'],
										 validation_data=self.datagen('val', 'classifier'),
										 validation_steps=60,
										 callbacks=self.callbacks(weights_file),
										 workers=1)


		# Replace input layers
		new_input = Input(shape=(self.P['latent_dim'],))
		l = self.classifier.layers[-1](new_input)
		new_classifier = Model(new_input, l)
		new_classifier.compile(optimizer='adam', loss='mse')
		json_string = new_classifier.to_json()
		with open(json_file, 'w') as f:
			f.write(json_string)

		new_classifier.save(weights_file)
		loss_history = np.array(classifier_history.history["loss"])
		np.savetxt(loss_file, loss_history, delimiter=",")

		self.classifier = new_classifier
		new_classifier.summary()
		return

	def callbacks(self, weights_file):
		from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard, EarlyStopping
		import time

		callbacks = []
		callbacks.append(ModelCheckpoint(filepath=weights_file, verbose=1, monitor='val_loss', save_best_only=True))
		callbacks.append(EarlyStopping(monitor='val_loss',patience=5))
		callbacks.append(TensorBoard(log_dir='logs/{}'.format(time.strftime("%Y%m%d_%H%M%S")), write_graph=True, write_images=True, histogram_freq=0))
		callbacks.append(LearningRateScheduler(lambda epoch: 0.001 if epoch < 75 else 0.0001))
		return callbacks


	def datagen(self, dm, mode):
		# mode == vbae  | classifier
		# dm   == train | test | val
		from dataiter import Iterator

		n = 60 * 5 * 4 * 4 * 7 * 9 # nT * shapes * rotations
		generator      = Iterator(path.join('data', dm),
								  mode, n, self.P['nT'],
								  (self.P['im_height'], self.P['im_width']))

		return generator
