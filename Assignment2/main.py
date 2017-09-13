# coding: utf-8

import keras
from keras.datasets import cifar10

import tensorflow as tf

from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.layers import Conv2D, MaxPooling2D, Flatten, Input, Activation, Dense, merge
from keras.layers.merge import Maximum, Add

from custom_layers import Dropout, BatchNorm
from keras.models import Model
from keras.callbacks import TensorBoard

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
keras.backend.set_session(sess)


def preprocess_data():
	num_classes = 10
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()
	x_train = x_train.astype('float')
	x_test = x_test.astype('float')
	x_train /= 255.0
	x_test /= 255.0
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)
	return (x_train, y_train), (x_test, y_test)


def block(model, n_channels, resnet=False, activation=None, dropout=None, batchnorm=True):
	skip = model
	if activation:
		if resnet:
			model = Conv2D(n_channels, (3, 3),padding='same')(model)
		else:
			model = Conv2D(n_channels, (3, 3))(model)
		model = activation()(model)
		if dropout:
			model = Dropout(dropout)(model)
		if batchnorm:
			model = BatchNorm()(model)
		if resnet:
			model = Conv2D(n_channels, (3, 3))(model)
		else:
			model = Conv2D(n_channels, (3, 3))(model)
		if resnet:
			skip =  Conv2D(n_channels, (3, 3))(skip)
			model = Add()([model, skip])
		model = activation()(model)
		if dropout:
			model = Dropout(dropout)(model)
		if batchnorm:
			model = BatchNorm()(model)
	else:
		if resnet:
			model = Conv2D(n_channels, (3, 3),padding='same')(model)
		else:
			model = Conv2D(n_channels, (3, 3))(model)
		model = Activation('tanh')(model)
		if dropout:
			model = Dropout(dropout)(model)
		if batchnorm:
			model = BatchNorm()(model)
		if resnet:
			model = Conv2D(n_channels, (3, 3))(model)
		else:
			model = Conv2D(n_channels, (3, 3))(model)
		if resnet:
			skip =  Conv2D(n_channels, (3, 3))(skip)
			model = Add()([model, skip])
		model = Activation('tanh')(model)
		if dropout:
			model = Dropout(dropout)(model)
	try:
		model = MaxPooling2D(pool_size=2)(model)
	except:
		pass
	return model


def maxout_block(model, n_channels, resnet=False, activation=None, dropout=None, batchnorm=True):
	skip = model
	if activation:
		if resnet:
			left = Conv2D(n_channels, (3, 3), padding='same')(model)
			right = Conv2D(n_channels, (3, 3), padding='same')(model)
		else:
			left = Conv2D(n_channels, (3, 3))(model)
			right = Conv2D(n_channels, (3, 3))(model)
		model = Maximum()([left, right])
		model = activation()(model)
		if dropout:
			model = Dropout(dropout)(model)
		if batchnorm:
			model = BatchNorm()(model)
		if resnet:
			left2 = Conv2D(n_channels, (3, 3))(model)
			right2 = Conv2D(n_channels, (3, 3))(model)
		else:
			left2 = Conv2D(n_channels, (3, 3))(model)
			right2 = Conv2D(n_channels, (3, 3))(model)
		model = Maximum()([left2, right2])
		if resnet:
			skip =  Conv2D(n_channels, (3, 3))(skip)
			model = Add()([model, skip])
		model = activation()(model)
		if dropout:
			model = Dropout(dropout)(model)
		if batchnorm:
			model = BatchNorm()(model)
	else:
		if resnet:
			left = Conv2D(n_channels, (3, 3), padding='same')(model)
			right = Conv2D(n_channels, (3, 3), padding='same')(model)
		else:
			left = Conv2D(n_channels, (3, 3), padding='same')(model)
			right = Conv2D(n_channels, (3, 3), padding='same')(model)
		model = Maximum()([left, right])
		model = Activation('tanh')(model)
		if dropout:
			model = Dropout(dropout)(model)
		if batchnorm:
			model = BatchNorm()(model)
		if resnet:
			left2 = Conv2D(n_channels, (3, 3))(model)
			right2 = Conv2D(n_channels, (3, 3),)(model)
		else:
			left2 = Conv2D(n_channels, (3, 3))(model)
			right2 = Conv2D(n_channels, (3, 3))(model)
		model = Maximum()([left2, right2])
		if resnet:
			skip =  Conv2D(n_channels, (3, 3))(skip)
			model = Add()([model, skip])
		model = Activation('tanh')(model)
		if dropout:
			model = Dropout(dropout)(model)
	try:
		model = MaxPooling2D(pool_size=2)(model)
	except:
		pass
	return model


def bottleneck_block(model, n_channels, resnet=False, activation=None, dropout=None, batchnorm=True):
	skip = model
	
	model = Conv2D(n_channels, (1, 1))(model)
	model = activation()(model)
	model = Dropout(dropout)(model)
	
	model = Conv2D(n_channels, (3, 3))(model)
	model = activation()(model)
	model = Dropout(dropout)(model)
	
	model = Conv2D(n_channels*2, (1, 1))(model)
	skip = Conv2D(n_channels*2, (3,3))(skip)
	model = Add()([model, skip])
	
	model = activation()(model)
	model = Dropout(dropout)(model)
	
	model = BatchNorm()(model)
	try:
		model = MaxPooling2D(pool_size=2)(model)
	except:
		pass
	return model


def build_model(in_shape, n_classes, channels, module, hidden_vars, resnet, activation, dropout, batchnorm):
	model = Input(shape=in_shape)
	input_layer = model
	for channel in channels:
		model = module(model, channel, resnet, activation, dropout, batchnorm)
	model = Flatten()(model)
	if dropout:
		model = Dropout(dropout)(model)
	if activation:
		model = Dense(hidden_vars)(model)
	else:
		model = Dense(hidden_vars, activation='tanh')(model)
	if dropout:
		model = Dropout(dropout)(model)
	model = Dense(n_classes, activation='softmax')(model)
	assert(model != input_layer)
	network = Model(input_layer, model)
	opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)
	network.compile(loss='categorical_crossentropy',
			  optimizer=opt,
			  metrics=['accuracy'])
	return network


def train_model(hidden_vars):
	(x_train, y_train), (x_test, y_test) = preprocess_data()
	activation_functions = [LeakyReLU, PReLU, ELU]
	dropouts = [0.2, 0.4, 0.6]
	batch_sizes = [32, 64, 128]
	histories = []
	#part A
	print("PART A")
	model = build_model((32,32,3), 10, [32,64,128], block, hidden_vars, False, None, None, None)
	histories.append(model.fit(x_train, y_train, batch_size=16,epochs=100,validation_split=0.2,callbacks=[TensorBoard('./parta',write_grads=True,histogram_freq=25)]))
	print model.evaluate(x_test, y_test)
	#part B
	for i, activation in enumerate(activation_functions):
		print("PART B"+str(i))
		model = build_model((32,32,3), 10, [32,64,128], block, hidden_vars, False, activation, None, None)
		histories.append(model.fit(x_train, y_train, batch_size=16,epochs=100,validation_split=0.2,callbacks=[TensorBoard('./partb' + str(i),write_grads=True,histogram_freq=25)]))
		print model.evaluate(x_test, y_test)
	#part C
	for i, dropout in enumerate(dropouts):
		print("PART C"+str(i))
		model = build_model((32,32,3), 10, [32,64,128], block, hidden_vars, False, None, dropout, None)
		histories.append(model.fit(x_train, y_train, batch_size=16,epochs=100,validation_split=0.2,callbacks=[TensorBoard('./partc' + str(i),write_grads=True,histogram_freq=25)]))
		print model.evaluate(x_test, y_test)
	part C(d)
	print("PART C4")
	model = build_model((32,32,3), 10, [32,64,128], maxout_block, hidden_vars, False, None, None, None)
	histories.append(model.fit(x_train, y_train, batch_size=16,epochs=100,validation_split=0.2,callbacks=[TensorBoard('./partc4',write_grads=True,histogram_freq=25)]))
	print model.evaluate(x_test, y_test)
	#part D
	for i, bs in enumerate(batch_sizes):
		print("PART D"+str(i))
		model = build_model((32,32,3), 10, [32,64,128], block, hidden_vars, False, None, None, True)
		histories.append(model.fit(x_train, y_train, batch_size=bs,epochs=100,validation_split=0.2,callbacks=[TensorBoard('./partd' + str(i),write_grads=True,histogram_freq=25)]))
		print model.evaluate(x_test, y_test)
	#part E(a)
	print("PART E1")
	model = build_model((32,32,3), 10, [32,64,128], block, hidden_vars, True, ELU, 0.3, True)
	histories.append(model.fit(x_train, y_train, batch_size=16,epochs=100,validation_split=0.2,callbacks=[TensorBoard('./parte1',write_grads=True,histogram_freq=25)]))
	print model.evaluate(x_test, y_test)
	#part E(b)
	print("PART E2")
	model = build_model((32,32,3), 10, [32,64,128], bottleneck_block, hidden_vars, True, ELU, 0.3, True)
	histories.append(model.fit(x_train, y_train, batch_size=16,epochs=100,validation_split=0.2,callbacks=[TensorBoard('./parte2',write_grads=True,histogram_freq=25)]))
	print model.evaluate(x_test, y_test)


if __name__ == "__main__":
	train_model(512)
