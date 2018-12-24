import keras.backend as K
from keras.engine.topology import Layer
from keras.layers import Conv2D, BatchNormalization, LeakyReLU

class ConvBnLRelu(object):
    def __init__(self, filters, kernelSize, strides=1):
        self.filters = filters
        self.kernelSize = kernelSize
        self.strides = strides
    def __call__(self, net):
        net = Conv2D(self.filters, self.kernelSize,
                     strides=self.strides, padding='same')(net)
        net = BatchNormalization()(net)
        net = LeakyReLU()(net)
        return net

class SampleLayer(Layer):
    def __init__(self, **kwargs):
        self.beta = 26
        super(SampleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # save the shape for distribution sampling
        self.shape = input_shape[0]
        super(SampleLayer, self).build(input_shape) # needed for layers

    def call(self, x):
        mean     = x[0]
        stddev   = x[1]
        capacity = 20
        # kl divergence:
        latent_loss = -0.5 * K.mean(1 + stddev
                            - K.square(mean)
                            - K.exp(stddev))
        latent_loss = self.beta * K.abs(latent_loss - K.cast(capacity, 'float32')/self.shape[1])
        self.add_loss(latent_loss)
        epsilon = K.random_normal(shape=(self.shape[1],), mean=0., stddev=1.)
        return mean + K.exp(stddev) * epsilon

    def compute_output_shape(self, input_shape):
        return input_shape[0]
