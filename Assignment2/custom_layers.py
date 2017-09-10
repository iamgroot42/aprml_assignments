from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

# Base template (writing your own layer) from  https://keras.io/layers/writing-your-own-keras-layers/

class Dropout(Layer):

	def __init__(self, output_dim, **kwargs):
		self.output_dim = output_dim
		super(Dropout, self).__init__(**kwargs)

	def build(self, input_shape):
		self.kernel = self.add_weight(name='kernel', 
									  shape=(input_shape[1], self.output_dim),
									  initializer='uniform',
									  trainable=True)
		super(Dropout, self).build(input_shape)

	def call(self, x):
		return K.dot(x, self.kernel)

	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.output_dim)


class BatchNorm(Layer):

	def __init__(self, output_dim, **kwargs):
		self.epsilon = 0.0001
		super(BatchNorm, self).__init__(**kwargs)

	def build(self, input_shape):
		super(BatchNorm, self).build(input_shape)

	def call(self, x):
		mean = K.mean(x, axis=0)
		variance = K.var(x, axis=0)
		return (x - mean) / K.sqrt(variance + self.epsilon)

	def compute_output_shape(self, input_shape):
		return input_shape
