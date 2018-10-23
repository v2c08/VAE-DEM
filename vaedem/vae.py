import os 
import keras
import numpy as np
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Concatenate, LSTM, RepeatVector, Dense, Lambda, Layer, Add, Multiply, Bidirectional, TimeDistributed, InputSpec, GlobalAveragePooling1D, MaxPooling1D
from keras.layers.merge import Concatenate
from keras.callbacks import TensorBoard, Callback
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras import metrics
from keras.optimizers import Adam
from keras import objectives
from keras.utils import plot_model
from datetime import datetime
from gen_data import get_data
from keras.layers.merge import concatenate

# todo implement fit_generator()
# todo wrapper for fit/train
# todo FE layer
# todo FE hyperparameter selection

# Issue 1 - When to concatenate actions & obs:
# 		    Should policy decoder return a state representation?
#		    Or concat(env(v),v)?
#           fit_generator does not handle this

# Issue 2 - Incremental training is probably not the best solution
#			FeLayer could be used to joinly optimize Φ, ΘPD, ΘSD,
#			removing dependence on max-entropy RL approach
# 			but would force the 'self consistency' constraint onto
#			ΘPD-Φ rather than ΘSD-Φ. ΘPD intercts with the environment
# 			and we can place meaningful priors here



"""
Definitions:
x	   = input data (nBatches,nT,observation_dime + action_dim)
z 	   = latent representation of the input data
p(x)   = world
p(z)   = brain
q(z|x) = encoding of observaton into latent space
p(x|z) = decoding of latent variable into an prediction
"""

""" Custom Layer Definitions """
class KLLayer(Layer):
	"""
		Implements loss function (KL divergence) as a layer
	"""
	def __init__(self, *args, **kwargs):
		self.is_placeholder = True
		self.use_reconstruction_loss = P['use_rec_loss']
		super(KLLayer, self).__init__(*args, **kwargs)

	def call(self, inputs):
		# Inputs = (true vals, predicted vals)
		mu, log_sigma = inputs
		# get kl
		kl_batch = - .5 * K.sum(1 + log_sigma - K.square(mu) - K.exp(log_sigma), axis=-1)
		if self.use_reconstruction_loss:
			# get cross entropy - ('normal vae training' from paper)
			xentropy = K.mean(metrics.mean_squared_error(self.target, log_sigma), axis=1)
			# add kl&xent to model loss
			self.add_loss(K.mean(kl_batch+xentropy), inputs=inputs)
			return inputs #not used
		else:
			# add kl to model loss
			self.add_loss(K.mean(kl_batch), inputs=inputs)
			return inputs # not used


""" Misc / Logging Stuff """

def log_dir():
	""" Datetime for logging dir """
	date= datetime.now()
	r  = '{0}_{1:02d}_{2:02d}_'.format(date.year, date.month, date.day)
	r += '{0:02d}_{1:02d}_{2:02d}'.format(date.hour, date.minute, date.second)
	return r

def roll(x):
	"""	Shift 'time' forward by 1 for test data	"""
	shifted = np.zeros(shape=x.shape)
	shifted[:, :-1, :] = x[0:, 1:, 0:]
	return shifted


""" Environment & Parameters """
# Data Locations
WEIGHTS_DIR  = os.path.join(os.getcwd(),'weights')
RESULTS_DIR  = os.path.join(os.getcwd(),'results_pedestrian')
DATA_DIR     = os.path.join(os.getcwd(),'data\\')

t_files = sorted(os.listdir(DATA_DIR))
nT = 41
batch_size = 1
 
image_data, camxloc, camyloc, objxrot, objyrot = get_data(1, nT, 'train', DATA_DIR, (128,160), 3)

print(next(image_data))
print(next(image_data)[0].shape)
print(camxloc.shape)
print(objxrot.shape)

image_dim = np.prod((128,160,3))
cam_dim   = (3)
obj_dim   = (2)


P = {
    'nT': nT,
    'image_dim': image_dim,
    'cam_dim': cam_dim,
    'obj_dim': obj_dim,
    'batch_size': batch_size,
    'latent_dim': 8,
    'epochs': 100,
    'optimizer': Adam(lr=0.001),
	'encoder_hidden_units':128,
	'decoder_hidden_units':128,
    'decoder_dense_units': 64,
    'policy_hidden_units': (400,300,200),
	'policy_obs_units': 64,
	'policy_action_units': 4,
    'feature_rep_size': 100,
	'use_rec_loss':False,
	'verbose': True
}


"""		  	  Models  			"""
K.set_learning_phase(1) #set learning phase

# #  State Encoder - qφ(z | τ ) # #

def encoder():
    
    image_inputs = Input(shape=(P['nT'],P['image_dim']), name='image_inputs')
    cam_inputs   = Input(shape=(P['nT'],P['cam_dim']), name='cam_inputs')
    obj_inputs   = Input(shape=(P['nT'],P['obj_dim']), name='obj_inputs')
    
    # merge pixel representation and constraints
    inputs = concatenate([image_inputs, cam_inputs, obj_inputs])
    
    # Use informtion from the past & future
    h = Bidirectional(LSTM(P['encoder_hidden_units'],return_sequences=True))(inputs)

    # Mean pooling on temporal axis (compression)
    h = GlobalAveragePooling1D()(h)
    
    # mean net output
    z_mean = Dense(P['latent_dim'], name='zmean')(h)
    z_log_sigma = Dense(P['latent_dim'], name='zlogsigma')(h)
    z_mean, z_log_sigma = KLLayer()([z_mean, z_log_sigma])
    z = Lambda(sampling, output_shape=(P['latent_dim'],), name='z')([z_mean, z_log_sigma])
    
    z = concatenate([RepeatVector(P['nT'])(z), cam_inputs, obj_inputs])    
    
    return image_inputs, cam_inputs, obj_inputs, z_mean, z_log_sigma, z
    
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


image_inputs, cam_inputs, obj_inputs, z_mean, z_log_sigma, z = encoder()

encoder = Model([image_inputs, cam_inputs, obj_inputs], z, name='encoder')


encoder.summary()
plot_model(encoder, to_file='encoder.png', show_shapes=True)

# #  State Decoder - pθ(τ |z )  # #

def decoder():
    
    
    out_dim = P['image_dim']+P['cam_dim']+P['obj_dim']
    print(out_dim)
    image_out = Dense(P['image_dim'], activation='sigmoid')
    cam_out = Dense(P['cam_dim'], activation='sigmoid')
    obj_out = Dense(P['obj_dim'], activation='sigmoid')
    
    latent_inputs = Input(shape=(P['latent_dim'],), name='z_samples')
    z = RepeatVector(P['nT'])(latent_inputs)
    decoder_h = LSTM(P['latent_dim'], return_sequences=True)(z)
    s_hat = LSTM(32, return_sequences=True, activation='linear')(decoder_h)
    return latent_inputs, s_hat
    
    
latent_inputs, s_hat = decoder()
decoder = Model(latent_inputs, s_hat, name='state_decoder')
decoder.summary()
plot_model(decoder, to_file='decoder.png', show_shapes=True)

# VAE
# define cvae and encoder models
cvae = Model([image_inputs, cam_inputs, obj_inputs], decoder(z))
encoder = Model([image_inputs, cam_inputs, obj_inputs], z)


vae.fit([obs_data, shifted_obsdata],
        shuffle=True,
        epochs=P['epochs'],
        validation_split=0.1,
        batch_size=P['batch_size'],
        callbacks=[TensorBoard(log_dir='/dumps/seq_model_' + log_dir())])


encoder.save('encoder.h5')
decoder.save('decoder.h5')
policy.save('policy.h5')
policy.save('vae.h5')
