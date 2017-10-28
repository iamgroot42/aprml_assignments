# coding: utf-8

# For headless machines
import matplotlib
matplotlib.use('Agg')

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import os

from load_data import load, unlabelled_data

# Don't hog GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)


def dictionary_learning(X_train, lambda_val = 1, k = 128, learning_rate=1):
    X = tf.placeholder("float", X_train.shape)

    D = tf.placeholder("float", (X_train.shape[1], k), name='D')
    R = tf.placeholder("float", (k, X_train.shape[0]), name='R')
    
    D_weights = tf.Variable(tf.random_normal([X_train.shape[1], k]))
    R_weights = tf.Variable(tf.random_normal([k, X_train.shape[0]]))

    minimization_term = tf.norm(X - tf.transpose(tf.matmul(D_weights, R_weights)), ord='fro', axis=(0,1))
    regularization_term = lambda_val * tf.cast(tf.count_nonzero(R_weights), tf.float32)

    # Normal MSE loss for autoencoder:
    loss = minimization_term + regularization_term
    loss /= X_train.shape[0]
    optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    
    R_new = optimizer.minimize(loss, var_list=[R_weights])
    D_new = optimizer.minimize(loss, var_list=[D_weights])
    
    init = tf.global_variables_initializer()
    sess.run(init)
    
    num_epochs = 200

    # Train model
    Dw = None
    R_train = None
    errors = []
    for i in range(num_epochs):

        # Update R
	_, R_train = sess.run([R_new, R_weights],feed_dict={X: X_train})
        
        # Update D
        _, l, Dw = sess.run([D_new, loss, D_weights], feed_dict={X: X_train})
	errors.append(l)
        print("Training loss (MSE) : %.4f" % (l))

    plt.clf()
    plt.plot(1 + np.arange(num_epochs), errors, label='Training loss')
    plt.title('Training loss v/s Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    plt.savefig('dict_training.png')

    return Dw, R_train.T

(X_train, Y_train), (X_test, Y_test), cross_val_indices = load()

flat_X = X_train[cross_val_indices[0]]
flat_X = flat_X.reshape((flat_X.shape[0], np.prod(flat_X.shape[1:])))
test_flat_X = X_test.reshape((X_test.shape[0], np.prod(X_test.shape[1:])))


def find_reprs(X_train, D_train, lambda_val = 1, k=128, learning_rate=1):
    X = tf.placeholder("float", X_train.shape)

    D = tf.placeholder("float", D_train.shape)
    R = tf.placeholder("float", (k, X_train.shape[0]))
    
    R_weights = tf.Variable(tf.random_normal([k, X_train.shape[0]]))
    
    minimization_term = tf.norm(X - tf.transpose(tf.matmul(D, R_weights)), ord='fro', axis=(0,1))
    regularization_term = lambda_val * tf.cast(tf.count_nonzero(R_weights), tf.float32)

    # Normal MSE loss for autoencoder:
    loss = minimization_term + regularization_term
    loss /= X_train.shape[0]
    optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    
    R_new = optimizer.minimize(loss, var_list=[R_weights])
    
    init = tf.global_variables_initializer()
    sess.run(init)
    
    num_epochs = 50

    # Train model
    R_train = None
    for i in range(num_epochs):

        # Update R
        _, l, R_train = sess.run([R_new, loss, R_weights],feed_dict={X: X_train, D: D_train})
        print("Representation finding loss (MSE) : %.4f" % (l))

    return R_train.T

D, features_train = dictionary_learning(flat_X, 0.01, 100, 200)
features_test = find_reprs(test_flat_X, D, 0.01, 100, 100)
#features_test = np.linalg.pinv(test_flat_X, D.T)

print features_test.shape
print features_train.shape

def eucledian_prediction(X_tr, Y_tr, X_te, Y_te):
	correct = 0
	for i, test in enumerate(X_te):
		distances = []
		for point in X_tr:
			distances.append(np.linalg.norm(test - point))
		label = Y_tr[np.argmin(distances)]
		if np.argmax(label) == np.argmax(Y_te[i]):
			correct += 1
	accuracy  = correct / (1.0 * Y_te.shape[0])
	return accuracy


print("Dictionary-learning features based L2 accuracy:",eucledian_prediction(features_train, Y_train, features_test, Y_test))
	

if not os.path.exists("atoms"):
        os.makedirs("atoms")

for i, atom in enumerate(D.T):
        vis_atom = (atom.reshape((96, 96, 3)) * 255).astype('uint8')
        plt.clf()
        plt.imshow(vis_atom)
        plt.savefig("atoms/" + str(i+1) + "atom.png")

