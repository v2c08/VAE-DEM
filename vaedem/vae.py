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
from dataset import Dataset
import gym
from sampler import Sampler
from dataset import Dataset
from datetime import datetime


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
		self.target = Input(shape=(P['nT'], P['obs_dim']))
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

#sampler = Sampler(env_name, n_envs, envs=None)
env_name = 'MountainCarContinuous-v0'


env = gym.make(env_name)
lambda_env = lambda : gym.make(env_name) #todo

obs_dim = len(env.observation_space.sample())
action_dim = len(env.observation_space.sample())

nT = 20
batch_size = 50


dataset = Dataset(n_envs=1, n_batches=batch_size, nT=nT, obs_dim=obs_dim, action_dim=action_dim, data_path=None)
obs_data, action_data, combined_data = dataset.generate(env)
shifted_obsdata = roll(obs_data)


P = {
    'nT': nT,
    'obs_dim': obs_dim,
	'action_dim':action_dim,
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
	inputs = Input(shape=(P['nT'],P['obs_dim']), name='encoder_inputs')


	# Use informtion from the past & future
	h = Bidirectional(LSTM(P['encoder_hidden_units'],return_sequences=True))(inputs)
	# Mean pooling on temporal axis (compression)
	h = GlobalAveragePooling1D()(h)
	#h = MaxPooling1D(pool_length=P['nT'], name='max_pool')(h)

	# mean net output
	z_mean = Dense(P['latent_dim'], name='zmean')(h)
	z_log_sigma = Dense(P['latent_dim'], name='zlogsigma')(h)
	z_mean, z_log_sigma = KLLayer()([z_mean, z_log_sigma])
	z = Lambda(sampling, name='z')([z_mean, z_log_sigma])
	return inputs, z_mean, z_log_sigma, z

def sampling(args):
	"""
	lambda function for sampling latents
	z=μ(x)+Σ^.5(x)ϵ
	Can't perform backprop on a sampling operation
	Use eps to reparameterise z to be differentiable
	"""
	z_mean, z_log_sigma = args
	epsilon = K.random_normal(shape=(P['batch_size'],P['latent_dim']), mean=0., stddev=1)
	return z_mean + z_log_sigma * epsilon


inputs, z_mean, z_log_sigma, z = encoder()
encoder = Model(inputs, z, name='encoder')
encoder.summary()
plot_model(encoder, to_file='encoder.png', show_shapes=True)

# #  State Decoder - pθ(τ |z )  # #

def decoder():

	latent_inputs = Input(shape=(P['latent_dim'],), name='z_samples')
	z = RepeatVector(P['nT'])(latent_inputs)
	h = Dense(P['decoder_dense_units'], activation='relu')(z)
	h_zmean  = Dense(P['obs_dim'])(h)
	h_zsigma = Dense(P['obs_dim'])(h)
	s_hat = LSTM(P['obs_dim'], return_sequences=True, activation='linear')(h, [h_zmean, h_zsigma])

	return latent_inputs, s_hat


latent_inputs, s_hat = decoder()
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
