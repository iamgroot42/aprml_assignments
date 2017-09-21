from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

# Base template (writing your own layer) from  https://keras.io/layers/writing-your-own-keras-layers/

class Dropout(Layer):

	def __init__(self, probability, **kwargs):
		self.probability = probability
		assert(probability>=0 and probability<=1)
		super(Dropout, self).__init__(**kwargs)

	def build(self, input_shape):
		super(Dropout, self).build(input_shape)

	def call(self, x):
		return K.in_train_phase(K.dropout(x, self.probability),x)

	def compute_output_shape(self, input_shape):
		return input_shape


class BatchNorm(Layer):

	def __init__(self, **kwargs):
		self.epsilon = K.epsilon()
		super(BatchNorm, self).__init__(**kwargs)

	def build(self, input_shape):
		super(BatchNorm, self).build(input_shape)

	def call(self, x):
		mean = K.mean(x, axis=0)
		variance = K.var(x, axis=0)
		return K.in_train_phase((x - mean) / K.sqrt(variance + self.epsilon),x)

	def compute_output_shape(self, input_shape):
		return input_shape
