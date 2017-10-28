# coding: utf-8

# For headless machines
import matplotlib
matplotlib.use('Agg')

import numpy as np
import keras
import tensorflow as tf
import itertools    

from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.layers import Conv2D, MaxPooling2D, Flatten, Input, Dense, Activation, Dropout, Reshape, UpSampling2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2
from keras.initializers import he_normal

from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from keras.applications.resnet50 import ResNet50

from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt

from load_data import load, unlabelled_data


# Don't hog GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
keras.backend.set_session(sess)

def autoencoder(X_train):
    input_layer = Input(shape=(96,96,3))
    
    # Block 1
    encoder = Conv2D(32, (3, 3), border_mode='same', init=he_normal())(input_layer)
    encoder = PReLU()(encoder)
    encoder = MaxPooling2D()(encoder)

    # Block 2
    encoder = Conv2D(64, (3, 3), border_mode='same', init=he_normal())(encoder)
    encoder = PReLU()(encoder)
    encoder = MaxPooling2D()(encoder)
    
    # Block 3
    encoder = Conv2D(128, (3, 3), border_mode='same', init=he_normal())(encoder)
    encoder = PReLU()(encoder)
    encoder = MaxPooling2D()(encoder)

    # Block 1 decoder
    decoder = Conv2DTranspose(128, (3,3), border_mode='same', init=he_normal())(encoder)
    decoder = PReLU()(decoder)
    decoder = UpSampling2D()(decoder)

    # Block 2 decoder
    decoder = Conv2DTranspose(64, (3,3), border_mode='same', init=he_normal())(decoder)
    decoder = PReLU()(decoder)
    decoder = UpSampling2D()(decoder)

    # Block 3 decoder
    decoder = Conv2DTranspose(32, (3,3), border_mode='same', init=he_normal())(decoder)
    decoder = PReLU()(decoder)
    decoder = UpSampling2D()(decoder)
    decoder = Conv2D(3, (1,1), border_mode='same', init=he_normal())(decoder)
    decoder = Activation('sigmoid')(decoder)

    autoencoder = Model(input_layer, decoder)
    opt = keras.optimizers.Adadelta()
    autoencoder.compile(optimizer=opt, loss='mean_squared_error')
 
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.0001, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0005, patience=3, verbose=1)
    
    autoencoder.fit(X_train, X_train, epochs=15, batch_size=4,
                    validation_split=0.2, 
                    callbacks=[reduce_lr, early_stop])
    encoder = Model(input_layer, encoder)
    return autoencoder, encoder, input_layer

def reconstruct_image(vae, images, index=0):
    x_test = vae.predict(images)
    x_test = np.uint8(x_test * 255)
    plot_image = np.concatenate((np.uint8(images[index]*255), x_test[index]), axis=1)
    imgplot = plt.imshow(plot_image)

(X_train, Y_train), (X_test, Y_test), cross_val_indices = load()
X_unlabelled = unlabelled_data()

# aec, enc, inp = autoencoder(X_unlabelled)

# aec.save("part1_aec")
# enc.save("part1_enc")

aec = load_model("part1_aec")
enc = load_model("part1_enc")

#reconstruct_image(aec, X_unlabelled[:40], 0)
#plt.show()
#plt.savefig('reconstruction.png')
#exit()

def adapt_autoencoder(inp, enc, dense_only=True):
    encoder = Flatten()(enc)
    encoder = Dense(300, activation='relu')(encoder)
    encoder = Dense(10, activation='softmax')(encoder)
    whole_model = Model(inp, encoder)
    
    if dense_only:
        for layer in whole_model.layers:
            if "dense" not in layer.name:
                layer.trainable = False

    opt = keras.optimizers.Adadelta(0.1)
    whole_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.0001, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.005, patience=10, verbose=1)
    
    whole_model.fit(X_train, Y_train, epochs=50, batch_size=16,
                    validation_split=0.2,
                    callbacks=[reduce_lr, early_stop])
    print("\nTest Accuracy: ",whole_model.evaluate(X_test, Y_test)[1])
    return whole_model

deep_final_model = adapt_autoencoder(enc.input, enc.output, False)
dense_only_final_model = adapt_autoencoder(enc.input, enc.output, True)

def upsample(X):
    import Image
    resized = []
    for x in X:
        img = Image.fromarray(np.uint8(x * 255))
        resized.append(np.asarray(img.resize((198, 198), Image.NEAREST)).astype('float32')/255.0)
    return np.array(resized)

X_train = upsample(X_train)
X_test = upsample(X_test)

def adapt_resnet(dense_only=True): 
	resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(198,198,3))
	if dense_only:
		for layer in resnet.layers:
			layer.trainable = False

	output = Flatten()(resnet.output)
	output = Dense(10)(output)
	output = Activation('softmax')(output)
	model = Model(resnet.input, output) 

	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.0001, verbose=1)
	early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0005, patience=3, verbose=1)

	opt = keras.optimizers.Adadelta(0.1)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

	model.fit(X_train, Y_train, epochs=50, batch_size=16,
                    validation_split=0.2,
                    callbacks=[reduce_lr, early_stop])
	print("\nTest Accuracy: ",model.evaluate(X_test, Y_test)[1])

adapt_resnet(False)
adapt_resnet(True)
