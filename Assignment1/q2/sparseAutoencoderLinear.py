# This piece of software is bound by The MIT License (MIT)
# Copyright (c) 2014 Siddharth Agrawal
# Code written by : Siddharth Agrawal
# Email ID : siddharth.950@gmail.com

# Modified by : Anshuman Suri (iamgroot42)

import matplotlib as mpl
mpl.use('Agg')

import numpy
import math
import time
import scipy.io
import scipy.optimize
import matplotlib.pyplot
import numpy as np
from sklearn import linear_model
from sklearn.externals import joblib


def extract_data(filename, num_images, IMAGE_SIZE=28):
	with open(filename) as bytestream:
		bytestream.read(16)
		buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
		data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
		data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
	return data


def extract_labels(filename, num_images):
	with open(filename) as bytestream:
		bytestream.read(8)
		buf = bytestream.read(1 * num_images)
		labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
	return labels


###########################################################################################
""" The Sparse Autoencoder Linear class """

class SparseAutoencoderLinear(object):

    #######################################################################################
    """ Initialization of Autoencoder object """

    def __init__(self, visible_size, hidden_size, rho, lamda, beta):
    
        """ Initialize parameters of the Autoencoder object """
    
        self.visible_size = visible_size    # number of input units
        self.hidden_size = hidden_size      # number of hidden units
        self.rho = rho                      # desired average activation of hidden units
        self.lamda = lamda                  # weight decay parameter
        self.beta = beta                    # weight of sparsity penalty term
        
        """ Set limits for accessing 'theta' values """
        
        self.limit0 = 0
        self.limit1 = hidden_size * visible_size
        self.limit2 = 2 * hidden_size * visible_size
        self.limit3 = 2 * hidden_size * visible_size + hidden_size
        self.limit4 = 2 * hidden_size * visible_size + hidden_size + visible_size
        
        """ Initialize Neural Network weights randomly
            W1, W2 values are chosen in the range [-r, r] """
        
        r = math.sqrt(6) / math.sqrt(visible_size + hidden_size + 1)
        
        rand = numpy.random.RandomState(int(time.time()))
        
        W1 = numpy.asarray(rand.uniform(low = -r, high = r, size = (hidden_size, visible_size)))
        W2 = numpy.asarray(rand.uniform(low = -r, high = r, size = (visible_size, hidden_size)))
        
        """ Bias values are initialized to zero """
        
        b1 = numpy.zeros((hidden_size, 1))
        b2 = numpy.zeros((visible_size, 1))

        """ Create 'theta' by unrolling W1, W2, b1, b2 """

        self.theta = numpy.concatenate((W1.flatten(), W2.flatten(),
                                        b1.flatten(), b2.flatten()))

    #######################################################################################
    """ Returns elementwise sigmoid output of input array """
    
    def sigmoid(self, x):
    
        return (1 / (1 + numpy.exp(-x)))

    #######################################################################################
    """ Returns the cost of the Autoencoder and gradient at a particular 'theta' """

    def sparseAutoencoderLinearCost(self, theta, input):
        """ Extract weights and biases from 'theta' input """
        
        W1 = theta[self.limit0 : self.limit1].reshape(self.hidden_size, self.visible_size)
        W2 = theta[self.limit1 : self.limit2].reshape(self.visible_size, self.hidden_size)
        b1 = theta[self.limit2 : self.limit3].reshape(self.hidden_size, 1)
        b2 = theta[self.limit3 : self.limit4].reshape(self.visible_size, 1)
        
        """ Compute output layers by performing a feedforward pass
            Computation is done for all the training inputs simultaneously """
        
        hidden_layer = self.sigmoid(numpy.dot(W1, input) + b1)
        output_layer = numpy.dot(W2, hidden_layer) + b2
        
        """ Estimate the average activation value of the hidden layers """
        
        rho_cap = numpy.sum(hidden_layer, axis = 1) / input.shape[1]
        
        """ Compute intermediate difference values using Backpropagation algorithm """
        
        diff = output_layer - input
        
        sum_of_squares_error = 0.5 * numpy.sum(numpy.multiply(diff, diff)) / input.shape[1]
        weight_decay         = 0.5 * self.lamda * (numpy.sum(numpy.multiply(W1, W1)) +
                                                   numpy.sum(numpy.multiply(W2, W2)))
        KL_divergence        = self.beta * numpy.sum(self.rho * numpy.log(self.rho / rho_cap) +
                                                    (1 - self.rho) * numpy.log((1 - self.rho) / (1 - rho_cap)))

	def L1_grad(X):
		return np.sign(X)

	def L2_grad(X):
		return 2 *X

	def Lelastic_grad(X):
		return 0.5 * L1_grad(X)  + 0.5 * L2_grad(X)

	def Ltrace_grad(X):
		U, s, V = numpy.linalg.svd(X, full_matrices=False)
		return np.dot(U, V)

	L1_error             = numpy.linalg.norm(W1, 1) + numpy.linalg.norm(W2, 1)
	L2_error             = numpy.linalg.norm(W1, 2) + numpy.linalg.norm(W2, 2)
	Lelastic_error       = 0.5 * L1_error + 0.5 * L2_error
	Ltrace_error         = numpy.sum(numpy.linalg.svd(W1)[1]) + numpy.sum(numpy.linalg.svd(W2)[1])
        cost                 = sum_of_squares_error + weight_decay + KL_divergence + L1_error
        
        KL_div_grad = self.beta * (-(self.rho / rho_cap) + ((1 - self.rho) / (1 - rho_cap)))
        
        del_out = diff
        del_hid = numpy.multiply(numpy.dot(numpy.transpose(W2), del_out) + numpy.transpose(numpy.matrix(KL_div_grad)), 
                                 numpy.multiply(hidden_layer, 1 - hidden_layer))
        
        """ Compute the gradient values by averaging partial derivatives
            Partial derivatives are averaged over all training examples """
            
        W1_grad = numpy.dot(del_hid, numpy.transpose(input))
        W2_grad = numpy.dot(del_out, numpy.transpose(hidden_layer))
        b1_grad = numpy.sum(del_hid, axis = 1)
        b2_grad = numpy.sum(del_out, axis = 1)
            
        W1_grad = W1_grad / input.shape[1] + self.lamda * W1
        W2_grad = W2_grad / input.shape[1] + self.lamda * W2
        b1_grad = b1_grad / input.shape[1]
        b2_grad = b2_grad / input.shape[1]
        
        """ Transform numpy matrices into arrays """
        
        W1_grad = numpy.array(W1_grad) + L1_grad(W1)
        W2_grad = numpy.array(W2_grad) + L1_grad(W2)
        b1_grad = numpy.array(b1_grad)
        b2_grad = numpy.array(b2_grad)
        
        """ Unroll the gradient values and return as 'theta' gradient """
        
        theta_grad = numpy.concatenate((W1_grad.flatten(), W2_grad.flatten(),
                                        b1_grad.flatten(), b2_grad.flatten()))
                                        
        return [cost, theta_grad]

	###########################################################################################
	""" Train a classification model over encodings """
	
    def trainMLP(self, theta, X_train, y_train, X_test, y_test):
	W1 = theta[self.limit0 : self.limit1].reshape(self.hidden_size, self.visible_size)
	W2 = theta[self.limit1 : self.limit2].reshape(self.visible_size, self.hidden_size)
	b1 = theta[self.limit2 : self.limit3].reshape(self.hidden_size, 1)
	b2 = theta[self.limit3 : self.limit4].reshape(self.visible_size, 1)
	
	logreg = linear_model.LogisticRegression(C=1e5)
	X_enc_train = self.sigmoid(numpy.dot(W1, X_train) + b1)
	X_enc_test = self.sigmoid(numpy.dot(W1, X_test) + b1)
	X_enc_train = np.swapaxes(X_enc_train, 0, 1)
	X_enc_test = np.swapaxes(X_enc_test, 0, 1)
	logreg.fit(X_enc_train, y_train)
	print logreg.score(X_enc_test, y_test)
	return logreg

###########################################################################################
""" Loads the dataset from ubtye files """

def loadDataset():

    """ Loads the dataset as a numpy array
        The dataset is originally read as a dictionary """
    X_test = extract_data("notMNIST-to-MNIST/data/t10k-images-idx3-ubyte", 1000 * 10)
    X_test = np.swapaxes(X_test.reshape((1000 * 10, 28 * 28 * 1)), 0, 1) / 255.0
    X_train = extract_data("notMNIST-to-MNIST/data/train-images-idx3-ubyte", 6000 * 10)
    X_train = np.swapaxes(X_train.reshape((6000 * 10, 28 * 28 * 1)), 0, 1) / 255.0
    Y_test =  extract_labels("notMNIST-to-MNIST/data/t10k-labels-idx1-ubyte", 1000 * 10)
    Y_train =  extract_labels("notMNIST-to-MNIST/data/train-labels-idx1-ubyte", 6000 * 10)
    return (X_train, Y_train), (X_test, Y_test)
    
###########################################################################################
""" Visualizes the obtained optimal W1 values as images """

def visualizeW1(opt_W1, vis_patch_side, hid_patch_side):

    """ Add the weights as a matrix of images """
    
    figure, axes = matplotlib.pyplot.subplots(nrows = hid_patch_side,
                                              ncols = hid_patch_side)
    
    """ Rescale the values from [-1, 1] to [0, 1] """
    
    opt_W1 = (opt_W1 + 1) / 2
    
    """ Define useful values """
    
    index  = 0
    limit = vis_patch_side * vis_patch_side
    for axis in axes.flat:
    
        """ Initialize image as array of zeros """
    
        img = numpy.zeros((vis_patch_side, vis_patch_side))
        
        """ Divide the rows of parameter values into image channels """
        
        img[:, :] = opt_W1[index, 0 : limit].reshape(vis_patch_side, vis_patch_side)
        
        """ Plot the image on the figure """
        
        image = axis.imshow(img, interpolation = 'nearest')
        axis.set_frame_on(False)
        axis.set_axis_off()
        index += 1
        
    """ Show the obtained plot """  
        
    matplotlib.pyplot.savefig('weights1.png')

###########################################################################################
""" Visualizes the obtained optimal W1 values as images """

def visualizeW2(opt_W2, vis_patch_side, hid_patch_side):

    """ Add the weights as a matrix of images """
    
    figure, axes = matplotlib.pyplot.subplots(nrows = hid_patch_side,
                                              ncols = hid_patch_side)
    
    """ Define useful values """
    
    index  = 0
    limit =  vis_patch_side * vis_patch_side
                                              
    for axis in axes.flat:
    
        """ Initialize image as array of zeros """
    
        img = numpy.zeros((vis_patch_side, vis_patch_side))
        
        """ Divide the rows of parameter values into image channels """
        
        img[:, :] = opt_W2[index, 0 : limit].reshape(vis_patch_side, vis_patch_side)
        
        """ Plot the image on the figure """
        
        image = axis.imshow(img, interpolation = 'nearest')
        axis.set_frame_on(False)
        axis.set_axis_off()
        index += 1
        
    """ Show the obtained plot """  
        
    matplotlib.pyplot.savefig('weights2.png')

###########################################################################################
""" Loads data, trains the Autoencoder and visualizes the learned weights """

def executeSparseAutoencoderLinear():

    """ Define the parameters of the Autoencoder """
    
    image_channels  = 1      # number of channels in the image patches
    vis_patch_side  = 28     # side length of sampled image patches
    hid_patch_side  = 20     # side length of representative image patches
    num_patches     = 10000  # number of training examples
    rho             = 0.035  # desired average activation of hidden units
    lamda           = 0.003  # weight decay parameter
    beta            = 5      # weight of sparsity penalty term
    max_iterations  = 050    # number of optimization iterations
    
    visible_size = vis_patch_side * vis_patch_side * image_channels # number of input units
    hidden_size  = hid_patch_side * hid_patch_side                  # number of hidden units
    
    """ Load the dataset and preprocess using ZCA Whitening """
    
    (xtr, ytr), (xte, yte) = loadDataset()
    
    """ Initialize the Autoencoder with the above parameters """
    
    encoder = SparseAutoencoderLinear(visible_size, hidden_size, rho, lamda, beta)
    
    """ Run the L-BFGS algorithm to get the optimal parameter values """
    
    opt_solution  = scipy.optimize.minimize(encoder.sparseAutoencoderLinearCost, encoder.theta, 
                                            args = (xtr,), method = 'L-BFGS-B', 
                                            jac = True, options = {'maxiter': max_iterations, 'disp': True})
    opt_theta     = opt_solution.x
    opt_W1        = opt_theta[encoder.limit0 : encoder.limit1].reshape(hidden_size, visible_size)
    opt_W2        = opt_theta[encoder.limit1 : encoder.limit2].reshape(visible_size, hidden_size)
    
    """ Visualize the obtained optimal weights """
    visualizeW1(opt_W1, vis_patch_side, hid_patch_side)
    visualizeW2(opt_W2, hid_patch_side, vis_patch_side)
    clf = encoder.trainMLP(opt_theta, xtr, ytr, xte, yte)
    joblib.dump(clf, 'MLP.pkl') 

if __name__ == "__main__":
    executeSparseAutoencoderLinear()
