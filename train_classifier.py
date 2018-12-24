import os
from models import SampleLayer
from keras.models import load_model, model_from_json
from gen_data import get_data
import numpy as np
from keras import backend as K
from keras import metrics
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Input, Dense, Flatten, TimeDistributed, Lambda, Reshape, Concatenate, Masking, Convolution2D, Activation, MaxPooling2D, Dropout, Embedding, Convolution1D, MaxPooling1D, LSTM, Bidirectional, GlobalAveragePooling1D, Conv2DTranspose, Conv3D, MaxPooling3D, Conv2D, Conv3DTranspose, UpSampling3D, GlobalAveragePooling2D, BatchNormalization, LeakyReLU, MaxPool2D, UpSampling2D
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard, EarlyStopping
import time
from keras.optimizers import Adam



DATA_DIR     = os.path.join(os.getcwd(),'data')
if not os.path.exists(DATA_DIR):    os.makedirs(DATA_DIR)


json_file = open('weights/encoder.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
encoder = model_from_json(loaded_model_json, custom_objects={'SampleLayer': SampleLayer})
encoder.load_weights("weights/encoder.h5")

for layer in encoder.layers:
    layer.trainable = False


train_batches = 100 #5 * 4 * 4 * 9 * 7
epochs = 1
nT = 60
nClasses = 9 # cprod (shapes x colours)
latent_dim = 8

train_generator = get_data(train_batches, nT, 'train', DATA_DIR, (64,64), 3, 'classifier')
validation_generator = get_data(train_batches, nT, 'val', DATA_DIR, (64,64), 3, 'classifier')

# colour classifier
latent_inputs  = Input(shape=(latent_dim,), name='classifier_inputs')
x = Dense(128)(latent_inputs)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(nClasses)(x)
classification_output = Activation('sigmoid')(x)

class_model = Model(latent_inputs, classification_output)

classifier = Model(encoder.input, class_model(encoder.output))

for layer in encoder.layers:
    layer.trainable = False


weights_file = os.path.join('weights/classifier.hdf5')  # where weights will be saved
json_file    = os.path.join('weights/classifier.json')
loss_file    = os.path.join('weights/loss.txt')

callbacks = []
callbacks.append(ModelCheckpoint(filepath=weights_file, verbose=1, monitor='loss', save_best_only=True))
#callbacks.append(EarlyStopping(monitor='val_loss',patience=5))
callbacks.append(TensorBoard(log_dir='logs/{}/{}'.format(latent_dim,time.strftime("%Y%m%d_%H%M%S")), write_graph=True, write_images=True, histogram_freq=0))
callbacks.append(LearningRateScheduler(lambda epoch: 0.001 if epoch < 75 else 0.0001))

classifier.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])
classifier_history = classifier.fit_generator(train_generator,
                                              epochs=epochs,
                                              steps_per_epoch=train_batches,
                                              validation_data=validation_generator,
                                              validation_steps=60,
                                              callbacks=callbacks,
                                              workers=1)


json_string = classifier.to_json()
with open(json_file, 'w') as f:
    f.write(json_string)

loss_history = np.array(classifier_history.history["loss"])
np.savetxt(loss_file, loss_history, delimiter=",")
