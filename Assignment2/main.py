from keras.datasets import cifar10

from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU, merge
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Input


def preprocess_data():
	num_classes = 10
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()
	x_train /= 255.0
	x_test /= 255.0
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)


def block(model, n_channels):
	model = Convolution2D(n_channels, 3, 3, activation='relu', input_shape=(1,28,28))(model)
	return model

def build_model(in_shape, n_classes, channels):
	model = Input(shape=in_shape)
	for channel in channels:
		model = block(channel)
	model.compile(optimizer='rmsprop',
          loss='categorical_crossentropy',
          metrics=['accuracy'])
	model.add(Flatten())
	model.add(Dense())
	model.add(Dense(n_classes))
	# z = merge([x, y], mode='sum') #ResNet
	return model


if __name__ == "__main__":
	print "main"
