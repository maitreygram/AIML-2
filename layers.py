import numpy as np

class FullyConnectedLayer:
	def __init__(self, in_nodes, out_nodes):
		# Method to initialize a Fully Connected Layer
		# Parameters
		# in_nodes - number of input nodes of this layer
		# out_nodes - number of output nodes of this layer
		self.in_nodes = in_nodes
		self.out_nodes = out_nodes
		# Stores the outgoing summation of weights * feautres 
		self.data = None

		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1,(in_nodes, out_nodes))	
		self.biases = np.random.normal(0,0.1, (1, out_nodes))
		###############################################
		# NOTE: You must NOT change the above code but you can add extra variables if necessary 

	def forwardpass(self, X):
		# print('Forward FC ',self.weights.shape)
		# Input
		# activations : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_nodes]
		# OUTPUT activation matrix		:[n X self.out_nodes]

		###############################################
		# print("FullyConnectedLayer forward X ", X.shape)
		# print("FullyConnectedLayer forward weights ", self.weights.shape)
		# print("FullyConnectedLayer forward biases ", self.biases.shape)
		# print()
		self.data = np.matmul(X,self.weights) + self.biases
		return sigmoid(self.data)
		# raise NotImplementedError
		###############################################
		
	def backwardpass(self, lr, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE
		delta = delta * derivative_sigmoid(self.data)
		new_delta = np.matmul(delta, self.weights.T)
		self.weights -= lr * np.matmul(activation_prev.T, delta)
		self.biases -= lr * np.sum(delta, axis=0).reshape(self.biases.shape)
		return new_delta
		###############################################

class ConvolutionLayer:
	def __init__(self, in_channels, filter_size, numfilters, stride):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for convolution layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer
		# numfilters  - number of feature maps (denoting output depth)
		# stride	  - stride to used during convolution forward pass
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		self.out_depth = numfilters
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

		# Stores the outgoing summation of weights * feautres 
		self.data = None
		
		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1, (self.out_depth, self.in_depth, self.filter_row, self.filter_col))	
		self.biases = np.random.normal(0,0.1,self.out_depth)
		

	def forwardpass(self, X):
		# print('Forward CN ',self.weights.shape)
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		n = X.shape[0]  # batch size
		self.data = np.zeros([n,self.out_depth,self.out_row,self.out_col])
		# INPUT activation matrix  		:[n X self.in_channels[0] X self.in_channels[1] X self.in_channels[2]]
		# OUTPUT activation matrix		:[n X self.outputsize[0] X self.outputsize[1] X self.numfilters]

		###############################################
		# TASK 1 - YOUR CODE HERE
		for i in range(self.out_depth):
			for j in range(self.out_row):
				for k in range(self.out_col):
					self.data[:,i,j,k] = np.sum(self.weights[i,:,:,:] *
												X[:,:,j*self.stride:j*self.stride+self.filter_row,k*self.stride:k*self.stride+self.filter_col])
			self.data[:,i,:,:] += self.biases[i]
		return sigmoid(self.data)
		# raise NotImplementedError
		###############################################

	def backwardpass(self, lr, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE
		# print("ConvolutionLayer delta ",delta.shape)
		# print("ConvolutionLayer activation_prev ",activation_prev.shape)
		# print("ConvolutionLayer self.weights ",self.weights.shape)
		# print("ConvolutionLayer self.data ",self.data.shape)
		# print("ConvolutionLayer self.biases ",self.biases.shape)
		new_delta = np.zeros([n,self.in_depth,self.in_row,self.in_col])
		delta = delta * derivative_sigmoid(self.data)

		self.biases -= np.sum(delta, axis=(0,2,3)).reshape(self.biases.shape)
		for i in range(self.out_depth):
			for j in range(self.in_depth):
				for k in range(self.filter_row):
					for l in range(self.filter_col):
						self.weights[i,j,k,l] -= lr * np.sum(delta[:,i,:,:] * activation_prev[:,j,k:self.in_row-self.filter_row+1+k:self.stride,l:self.in_col-self.filter_col+1+l:self.stride])

		for i in range(self.in_depth):
			for j in range(self.in_row):
				for k in range(self.in_col):
					delta_j1 = max(0,j-self.filter_row+1)
					delta_j2 = min(self.out_row,j+1)
					delta_k1 = max(0,k-self.filter_col+1)
					delta_k2 = min(self.out_col,k+1)

					wt_j1 = max(0,j-self.out_col+1)
					wt_j2 = min(self.filter_row,j+1)
					wt_k1 = max(0,k-self.out_col+1)
					wt_k2 = min(self.filter_col,k+1)

					delta_dash = delta[:,:,delta_j1:delta_j2,delta_k1:delta_k2]
					weights_dash = self.weights[:,i,wt_j1:wt_j2,wt_k1:wt_k2]
					new_delta[:,i,j,k] = np.sum(delta_dash * weights_dash)
		return new_delta
		# new_delta = np.matmul(delta, self.weights.T)
		# self.weights -= lr * np.matmul(activation_prev.T, delta)
		# self.biases -= lr * np.sum(delta, axis=0).reshape(self.biases.shape)
		# return new_delta
		# raise NotImplementedError
		###############################################
	
class AvgPoolingLayer:
	def __init__(self, in_channels, filter_size, stride):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for max_pooling layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer

		# NOTE: Here we assume filter_size = stride
		# And we will ensure self.filter_size[0] = self.filter_size[1]
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		self.out_depth = self.in_depth
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

	def forwardpass(self, X):
		# print('Forward MP ')
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		n = X.shape[0]  # batch size
		X_new = np.zeros([n,self.out_depth,self.out_row,self.out_col])
		# INPUT activation matrix  		:[n X self.in_channels[0] X self.in_channels[1] X self.in_channels[2]]
		# OUTPUT activation matrix		:[n X self.outputsize[0] X self.outputsize[1] X self.in_channels[2]]

		###############################################
		# TASK 1 - YOUR CODE HERE
		for j in range(self.out_row):
			for k in range(self.out_col):
				X_new[:,:,j,k] = np.mean(X[:,:,j*self.stride:j*self.stride+self.filter_row,k*self.stride:k*self.stride+self.filter_col],axis=(2,3))
		return X_new
		# raise NotImplementedError
		###############################################


	def backwardpass(self, alpha, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# activations_curr : Activations of current layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		n = activation_prev.shape[0] # batch size
		new_delta = np.zeros([n,self.in_depth,self.in_row,self.in_col])
		###############################################
		# TASK 2 - YOUR CODE HERE
		# print("AvgPoolingLayer delta ", delta.shape)
		# print("AvgPoolingLayer activation_prev ", activation_prev.shape)
		for j in range(self.out_row):
			for k in range(self.out_col):
				# print(delta.shape)
				# print(np.repeat(np.repeat(delta[:,:,j,k].reshape(n,self.in_depth,1,1),self.filter_row,axis=2),self.filter_col,axis=3).shape)
				new_delta[:,:,j*self.stride:j*self.stride+self.filter_row,k*self.stride:k*self.stride+self.filter_col] = np.repeat(np.repeat(delta[:,:,j,k].reshape(n,self.in_depth,1,1),self.filter_row,axis=2),self.filter_col,axis=3)
		return new_delta
		# raise NotImplementedError
		###############################################


# Helper layer to insert between convolution and fully connected layers
class FlattenLayer:
    def __init__(self):
        pass
    
    def forwardpass(self, X):
        self.in_batch, self.r, self.c, self.k = X.shape
        return X.reshape(self.in_batch, self.r * self.c * self.k)

    def backwardpass(self, lr, activation_prev, delta):
        return delta.reshape(self.in_batch, self.r, self.c, self.k)


# Helper Function for the activation and its derivative
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
	return sigmoid(x) * (1 - sigmoid(x))