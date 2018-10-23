import os
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import time
from gen_data import get_data
from keras import backend as K
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Input, Dense, Flatten, TimeDistributed, Lambda, Reshape, Concatenate
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam
from keras.backend import set_image_data_format,slice
import argparse

set_image_data_format('channels_last')

WEIGHTS_DIR  = os.path.join(os.getcwd(),'weights')
RESULTS_DIR  = os.path.join(os.getcwd(),'results')
DATA_DIR     = os.path.join(os.getcwd(),'data')

print(WEIGHTS_DIR)
print(RESULTS_DIR)

if not os.path.exists(WEIGHTS_DIR): os.makedirs(WEIGHTS_DIR)
if not os.path.exists(DATA_DIR):    os.makedirs(DATA_DIR)
if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)

weights_file = os.path.join(WEIGHTS_DIR, 'vaedem.hdf5')  # where weights will be saved
json_file    = os.path.join(WEIGHTS_DIR, 'vaedem.json')


# these are determined by the number of subdirectories under data
# dont change
train_batches = 1
test_batches  = 1
val_batches   = 1
nT 		    = 41
epochs      = 1

n_plot       = nT

target_size = (160,128)

# Get data & dimensions
image_generator, camloc, objrot      = get_data(train_batches, nT, 'train', DATA_DIR, target_size, 3)
#validation_generator = get_data(val_batches,  nT, 'val',   DATA_DIR, target_size, 3)
#test_generator       = get_data(test_batches, nT, 'test',  DATA_DIR, target_size, 3)



im_width, im_height, n_channels = 160, 128, 3

image_shape  = (im_width, im_height, n_channels)
image_inputs = Input(shape=(nT,)+image_shape)

obj_shape  = (3,)
obj_inputs = Input(shape=(nT,)+obj_shape)

cam_shape  = (2,)
cam_inputs = Input(shape=(nT,)+cam_shape)

inputs = Concatenate(image_inputs, obj_inputs)
inputs = Concatenate(inputs, cam_inputs)


# Use informtion from the past & future
h = Bidirectional(LSTM(64,return_sequences=True))(inputs)
# Mean pooling on temporal axis (compression)
h = GlobalAveragePooling1D()(h)
#h = MaxPooling1D(pool_length=P['nT'], name='max_pool')(h)

# mean net output
z_mean = Dense(P['latent_dim'], name='zmean')(h)
z_log_sigma = Dense(P['latent_dim'], name='zlogsigma')(h)
z_mean, z_log_sigma = KLLayer()([z_mean, z_log_sigma])
z = Lambda(sampling, name='z')([z_mean, z_log_sigma])

z_cond = concatenate([z, cam_inputs, obj_inputs])


def sampling(args):
	"""
	lambda function for sampling latents
	z=μ(x)+Σ^.5(x)ϵ
	Can't perform backprop on a sampling operation
	Use eps to reparameterise z to be differentiable
	"""
	z_mean, z_log_sigma = args
	epsilon = K.random_normal(shape=(P['batch_size'],P['latent_dim']), mean=0., stddev=1)
	return z_mean + K.exp(z_log_sigma/2) * epsilon

encoder = Model(inputs, z, name='encoder')
encoder.summary()
plot_model(encoder, to_file='encoder.png', show_shapes=True)

# #  State Decoder - pθ(τ |z )  # #

latent_inputs = Input(shape=(P['latent_dim'],), name='z_samples')
z = RepeatVector(P['nT'])(latent_inputs)
h = Dense(P['decoder_dense_units'], activation='relu')(z)
h_zmean  = Dense(P['obs_dim'])(h)
h_zsigma = Dense(P['obs_dim'])(h)
s_hat = LSTM(P['obs_dim'], return_sequences=True, activation='linear')(h, [h_zmean, h_zsigma])


decoder = Model(latent_inputs, s_hat, name='state_decoder')
decoder.summary()
plot_model(decoder, to_file='decoder.png', show_shapes=True)

# VAE
kl_layer = KLLayer(name='loss_layer')
y = kl_layer([inputs, decoder(encoder(inputs))]) # real & deconstructed inputs respectively
vae = Model([inputs, kl_layer.target], y)
vae.compile(optimizer=P['optimizer'],loss=None) # We have implemented loss in KLLayer
vae.summary()
plot_model(vae, to_file='vae.png', show_shapes=True)
































predictions = prednet(inputs)
model = Model(inputs=inputs, outputs=predictions)
model.compile(loss=extrap_loss, optimizer='adam')


model.summary()
plot_model(model, to_file='prednet_{}.png'.format(MODE), show_shapes=True)

lr_schedule = lambda epoch: 0.001 if epoch < 75 else 0.0001    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs

callbacks = [LearningRateScheduler(lr_schedule)]
callbacks.append(ModelCheckpoint(filepath=weights_file, verbose=1, monitor='val_loss', save_best_only=True))
print('Training')

history = model.fit_generator(train_generator,
                              epochs=epochs,
                              steps_per_epoch=train_batches,
                              validation_data=validation_generator,
                              validation_steps=val_batches,
                              callbacks=callbacks,
                              workers=1)
#TODO multiple workers speed things up but may shuffle data. Verify this.

if save_model:
    json_string = model.to_json()
    with open(json_file, 'w') as f:
        f.write(json_string)

X_hat = model.predict_generator(test_generator, steps=nT)

i = 0
j = 0

# Plot some predictions
aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]
plt.figure(figsize = (nT, 2*aspect_ratio))
gs = gridspec.GridSpec(2, nT)
gs.update(wspace=0., hspace=0.)

plot_save_dir = os.path.join(RESULTS_DIR, 'prediction_plots/')
#shutil.rmtree(plot_save_dir)
if not os.path.exists(plot_save_dir): os.mkdir(plot_save_dir)


mse_model = 0
mse_prev = 0
test_data = np.zeros_like(X_hat)
test_generator_2       = get_data(nT, 'test',  DATA_DIR, target_size, 3, MODE)



plot_idx = list(range(test_data.shape[0]))
for i in plot_idx:
    d = next(test_generator_2)[0]

    mse_model += np.mean( (d[1:] - X_hat[i, 1:])**2 )  # look at all timesteps except the first
    mse_prev += np.mean( (d[:-1] - d[1:])**2 )

    for t in range(nT):

        plt.subplot(gs[t])
        test_data[i,t,:,:,:] = d[t]
        image1 = test_data[i,t]
        plt.imshow(image1)
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t==0: plt.ylabel('Actual', fontsize=10)

        plt.subplot(gs[t + nT])
        image2 = X_hat[i,t]#.astype(np.uint8)
        plt.imshow(image2)
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t==0: plt.ylabel('Predicted', fontsize=10)

    plt.savefig(plot_save_dir + 'plot_' + str(i) + '.png')
    plt.clf()

f = open(RESULTS_DIR + 'prediction_scores.txt', 'w')
f.write("Model MSE: %f\n" % mse_model)
f.write("Previous Frame MSE: %f" % mse_prev)
f.close()
