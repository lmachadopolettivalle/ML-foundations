import pandas as pd
import logging
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import math

### Assignment Owner: Tian Wang

#######################################
#### Normalization


def feature_normalization(train, test):
	"""Rescale the data so that each feature in the training set is in
	the interval [0,1], and apply the same transformations to the test
	set, using the statistics computed on the training set.

	Args:
		train - training set, a 2D numpy array of size (num_instances, num_features)
		test  - test set, a 2D numpy array of size (num_instances, num_features)
	Returns:
		train_normalized - training set after normalization
		test_normalized  - test set after normalization

	"""
	num_instances = train.shape[0]
	num_features = train.shape[1]
	train = np.transpose(train)
	test = np.transpose(test)
	train_normalized = np.empty([num_features, num_instances])
	test_normalized = np.empty([num_features, num_instances])

	for i in xrange(num_features):
		small = train[i].min()
		large = train[i].max()
		if small == large:
			continue
		train_normalized[i] = (train[i]-small)/(large-small)
		test_normalized[i] = (test[i]-small)/(large-small)
	
	train_normalized = np.transpose(train_normalized)
	test_normalized = np.transpose(test_normalized)

	return train_normalized, test_normalized

########################################
#### The square loss function

def compute_square_loss(X, y, theta):
	"""
	Given a set of X, y, theta, compute the square loss for predicting y with X*theta

	Args:
		X - the feature vector, 2D numpy array of size (num_instances, num_features)
		y - the label vector, 1D numpy array of size (num_instances)
		theta - the parameter vector, 1D array of size (num_features)

	Returns:
		loss - the square loss, scalar
	"""
	loss = 0 #initialize the square_loss
	m = X.shape[0] #num_instances

	if len(theta.shape) == 1:
		theta = theta[:, np.newaxis]
	if len(y.shape) == 1:
		y = y[:, np.newaxis]

	A = np.dot(X, theta) - y

	loss = (1.0/m) * np.dot(np.transpose(A), A)

	return loss

########################################
### compute the gradient of square loss function
def compute_square_loss_gradient(X, y, theta):
	"""
	Compute gradient of the square loss (as defined in compute_square_loss), at the point theta.

	Args:
		X - the feature vector, 2D numpy array of size (num_instances, num_features)
		y - the label vector, 1D numpy array of size (num_instances)
		theta - the parameter vector, 1D numpy array of size (num_features)

	Returns:
		grad - gradient vector, 1D numpy array of size (num_features)
	"""
	m = X.shape[0]
	if len(theta.shape) == 1:
		theta = theta[:, np.newaxis]
	if len(y.shape) == 1:
		y = y[:, np.newaxis]

	grad = (2.0/m) * (np.dot(np.dot(np.transpose(X), X), theta) - np.dot(np.transpose(X), y))

	return grad.flatten()


###########################################
### Gradient Checker
#Getting the gradient calculation correct is often the trickiest part
#of any gradient-based optimization algorithm.  Fortunately, it's very
#easy to check that the gradient calculation is correct using the
#definition of gradient.
#See http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
def grad_checker(X, y, theta, epsilon=0.01, tolerance=1e-4):
	"""Implement Gradient Checker
	Check that the function compute_square_loss_gradient returns the
	correct gradient for the given X, y, and theta.

	Let d be the number of features. Here we numerically estimate the
	gradient by approximating the directional derivative in each of
	the d coordinate directions:
	(e_1 = (1,0,0,...,0), e_2 = (0,1,0,...,0), ..., e_d = (0,...,0,1)

	The approximation for the directional derivative of J at the point
	theta in the direction e_i is given by:
	( J(theta + epsilon * e_i) - J(theta - epsilon * e_i) ) / (2*epsilon).

	We then look at the Euclidean distance between the gradient
	computed using this approximation and the gradient computed by
	compute_square_loss_gradient(X, y, theta).  If the Euclidean
	distance exceeds tolerance, we say the gradient is incorrect.

	Args:
		X - the feature vector, 2D numpy array of size (num_instances, num_features)
		y - the label vector, 1D numpy array of size (num_instances)
		theta - the parameter vector, 1D numpy array of size (num_features)
		epsilon - the epsilon used in approximation
		tolerance - the tolerance error

	Return:
		A boolean value indicate whether the gradient is correct or not

	"""
	true_gradient = compute_square_loss_gradient(X, y, theta) #the true gradient
	num_features = theta.shape[0]
	approx_grad = np.zeros(num_features) #Initialize the gradient we approximate

	if len(theta.shape) == 1:
		theta = theta[:, np.newaxis]
	if len(y.shape) == 1:
		y = y[:, np.newaxis]
	m = X.shape[0]

	for i in xrange(num_features):
		e_i = np.zeros(num_features)[:, np.newaxis]
		e_i[i] = 1
		
		J_plus = compute_square_loss(X, y, theta + epsilon*e_i)
		J_minus = compute_square_loss(X, y, theta - epsilon*e_i)

		approx_grad[i] = (1.0/(2*epsilon)) * (J_plus - J_minus)

	if np.linalg.norm(true_gradient - approx_grad) <= tolerance:
		return True
	else:
		return False

#################################################
### Generic Gradient Checker
def generic_gradient_checker(X, y, theta, objective_func, gradient_func, epsilon=0.01, tolerance=1e-4):
	"""
	The functions takes objective_func and gradient_func as parameters. And check whether gradient_func(X, y, theta) returned
	the true gradient for objective_func(X, y, theta).
	Eg: In LSR, the objective_func = compute_square_loss, and gradient_func = compute_square_loss_gradient
	"""
	true_gradient = gradient_func(X, y, theta) #the true gradient
	num_features = theta.shape[0]
	approx_grad = np.zeros(num_features) #Initialize the gradient we approximate

	if len(theta.shape) == 1:
		theta = theta[:, np.newaxis]
	if len(y.shape) == 1:
		y = y[:, np.newaxis]
	m = X.shape[0]

	for i in xrange(num_features):
		e_i = np.zeros(num_features)[:, np.newaxis]
		e_i[i] = 1
		
		J_plus = objective_func(X, y, theta + epsilon*e_i)
		J_minus = objective_func(X, y, theta - epsilon*e_i)

		approx_grad[i] = (1.0/(2*epsilon)) * (J_plus - J_minus)

	if np.linalg.norm(true_gradient - approx_grad) <= tolerance:
		return True
	else:
		return False

	
####################################
#### Batch Gradient Descent
def batch_grad_descent(X, y, alpha=0.1, num_iter=1000, check_gradient=False):
	"""
	In this question you will implement batch gradient descent to
	minimize the square loss objective

	Args:
		X - the feature vector, 2D numpy array of size (num_instances, num_features)
		y - the label vector, 1D numpy array of size (num_instances)
		alpha - step size in gradient descent
		num_iter - number of iterations to run
		check_gradient - a boolean value indicating whether checking the gradient when updating

	Returns:
		theta_hist - store the history of parameter vector in iteration, 2D numpy array of size (num_iter+1, num_features)
		*for instance, theta in iteration 0 should be theta_hist[0], theta in iteration (num_iter) is theta_hist[-1]
		loss_hist - the history of objective function vector, 1D numpy array of size (num_iter+1)
	"""
	num_instances, num_features = X.shape[0], X.shape[1]
	theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
	loss_hist = np.zeros(num_iter+1) #initialize loss_hist
	theta = np.zeros(num_features) #initialize theta
	
	for i in xrange(num_iter+1):
		theta_hist[i] = theta
		loss_hist[i] = compute_square_loss(X, y, theta)

		if check_gradient:
			if not grad_checker(X, y, theta):
				print "Gradient calculation is incorrect!"
				return 0

		grad = compute_square_loss_gradient(X, y, theta)

		theta = theta - alpha * grad

	return theta_hist, loss_hist


####################################
###Q2.4b: Implement backtracking line search in batch_gradient_descent
###Check http://en.wikipedia.org/wiki/Backtracking_line_search for details
#TODO


###################################################
### Compute the gradient of Regularized Batch Gradient Descent
def compute_regularized_square_loss_gradient(X, y, theta, lambda_reg):
	"""
	Compute the gradient of L2-regularized square loss function given X, y and theta

	Args:
		X - the feature vector, 2D numpy array of size (num_instances, num_features)
		y - the label vector, 1D numpy array of size (num_instances)
		theta - the parameter vector, 1D numpy array of size (num_features)
		lambda_reg - the regularization coefficient

	Returns:
		grad - gradient vector, 1D numpy array of size (num_features)
	"""
	m = X.shape[0]
	if len(theta.shape) == 1:
		theta = theta[:, np.newaxis]
	if len(y.shape) == 1:
		y = y[:, np.newaxis]

	grad = (2.0/m) * (np.dot(np.dot(np.transpose(X), X), theta) - np.dot(np.transpose(X), y)) + 2 * lambda_reg * theta
	return grad.flatten()

###################################################
### Batch Gradient Descent with regularization term
def regularized_grad_descent(X, y, alpha=0.1, lambda_reg=1, num_iter=1000):
	"""
	Args:
		X - the feature vector, 2D numpy array of size (num_instances, num_features)
		y - the label vector, 1D numpy array of size (num_instances)
		alpha - step size in gradient descent
		lambda_reg - the regularization coefficient
		num_iter - number of iterations to run

	Returns:
		theta_hist - the history of parameter vector, 2D numpy array of size (num_iter+1, num_features)
		loss_hist - the history of loss function without the regularization term, 1D numpy array.
	"""
	num_instances, num_features = X.shape[0], X.shape[1]
	theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
	loss_hist = np.zeros(num_iter+1) #initialize loss_hist
	theta = np.zeros(num_features) #initialize theta

	for i in xrange(num_iter+1):
		theta_hist[i] = theta
		loss_hist[i] = compute_square_loss(X, y, theta)

		grad = compute_regularized_square_loss_gradient(X, y, theta, lambda_reg)

		theta = theta - alpha * grad

	return theta_hist, loss_hist


#############################################
## Visualization of Regularized Batch Gradient Descent
##X-axis: log(lambda_reg)
##Y-axis: square_loss

#############################################
### Stochastic Gradient Descent
def stochastic_grad_descent(X, y, alpha=0.1, lambda_reg=1, num_iter=4):
	"""
	In this question you will implement stochastic gradient descent with a regularization term

	Args:
		X - the feature vector, 2D numpy array of size (num_instances, num_features)
		y - the label vector, 1D numpy array of size (num_instances)
		alpha - string or float. step size in gradient descent
				NOTE: In SGD, it's not always a good idea to use a fixed step size. Usually it's set to 1/sqrt(t) or 1/t
				if alpha is a float, then the step size in every iteration is alpha.
				if alpha == "1/sqrt(t)", alpha = 1/sqrt(t)
				if alpha == "1/t", alpha = 1/t
		lambda_reg - the regularization coefficient
		num_iter - number of epochs (i.e number of times) to go through the whole training set

	Returns:
		theta_hist - the history of parameter vector, 3D numpy array of size (num_iter, num_instances, num_features)
		loss hist - the history of regularized loss function vector, 2D numpy array of size(num_iter, num_instances)
	"""
	num_instances, num_features = X.shape[0], X.shape[1]
	theta = np.ones(num_features) #Initialize theta

	theta_hist = np.zeros((num_iter, num_instances, num_features))  #Initialize theta_hist

	loss_hist = np.zeros(num_iter) #Initialize loss_hist
	loss_by_step = True
	if loss_by_step:
		loss_hist = np.zeros(num_iter * num_instances) #Initialize loss_hist

	calc_step = False
	if type(alpha) is not float:
		calc_step = True
	else:
		step_size = alpha

	def step(alpha, t):
		if alpha == "1/sqrt(t)":
			return 1.0 / math.sqrt(t)
		elif alpha == "1/t":
			return 1.0 / t
		else:
			return None


	t = 1
	for epoch in xrange(num_iter):
		if not loss_by_step:
			loss_hist[epoch] = compute_square_loss(X, y, theta)
		for j in xrange(num_instances):
			t += 1
			theta_hist[epoch][j] = theta
			grad = compute_regularized_square_loss_gradient(X, y, theta, lambda_reg)

			if calc_step:
				step_size = step(alpha, t)
			theta = theta - step_size * grad
			if loss_by_step:
				loss_hist[j] = compute_square_loss(X, y, theta)

	return theta_hist, loss_hist, loss_by_step

################################################
### Visualization that compares the convergence speed of batch
###and stochastic gradient descent for various approaches to step_size
##X-axis: Step number (for gradient descent) or Epoch (for SGD)
##Y-axis: log(objective_function_value) and/or objective_function_value

def main():
	BGD_stepsize = False
	RGD_lambda_search = False
	choose_deploy_theta = False
	SGD_stepsizes = True


	#Loading the dataset
	print('Loading the dataset')

	df = pd.read_csv('data.csv', delimiter=',')
	X = df.values[:,:-1]
	y = df.values[:,-1]

	print('Split into Train and Test')
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=10)

	print("Scaling all to [0, 1]")
	B = 1
	X_train, X_test = feature_normalization(X_train, X_test)
	X_train = np.hstack((X_train, B * np.ones((X_train.shape[0], 1))))  # Add bias term
	X_test = np.hstack((X_test, B * np.ones((X_test.shape[0], 1)))) # Add bias term
	
	if BGD_stepsize:
		alpha = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
		train_theta_hist = []
		train_loss_hist = []
		color = ['k', 'k', 'r', 'r', 'g', 'g', 'b', 'b']
		line = ['-', '-.', '-', '-.', '-', '-.', '-', '-.']

		for i in xrange(len(alpha)):
			theta_hist, loss_hist = batch_grad_descent(X_train, y_train, alpha=alpha[i])
			plt.plot(loss_hist, color=color[i], linestyle=line[i], label=str(alpha[i]))
		plt.yscale('log')
		plt.xlabel('Iteration')
		plt.ylabel('Square Loss')
		plt.title('BGD Loss vs. Iteration Step for various step sizes')
		plt.legend(title='Step sizes')
		plt.savefig('BGD Loss for several step sizes.pdf')
		plt.show()

	if RGD_lambda_search:
		lamb = [1e-7, 1e-5, 1e-3, 1e-2, 1e-1, 1, 10, 100] #original lamb used
		#lamb = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
		#lamb = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
		L = len(lamb)
		theta_star = [] #found in for loop below from loss_hist.min
		train_loss = np.zeros(L)
		test_loss = np.zeros(L)

		for i in xrange(L):
			l = lamb[i]
			train_theta_hist, train_loss_hist = regularized_grad_descent(X_train, y_train, alpha=0.01, lambda_reg=l)

			theta = train_theta_hist[np.argmin(train_loss_hist)]
			theta_star.append(theta)
			
			train_loss[i] = compute_square_loss(X_train, y_train, theta)
			test_loss[i] = compute_square_loss(X_test, y_test, theta)
		

		if choose_deploy_theta:
			ind = np.argmin(test_loss)
			print "Choose following value of theta for deployment:\n", theta_star[ind]
			print "which corresponds to lambda_reg = ", lamb[ind]
			print "and to a test loss = ", test_loss[ind]
			print test_loss
		else:
			plt.plot(lamb, train_loss, label='Training Loss')
			plt.plot(lamb, test_loss, label='Test Loss')
			plt.xscale('log')
			plt.xlabel('Lambda reg.')
			plt.ylabel('Square Loss without reg.')
			plt.title('RGD Loss vs. Lambda reg. for training and test sets')
			plt.legend()
			#plt.savefig('RGD Lambda Factor-3.pdf')
			plt.show()
	
	if SGD_stepsizes:
		lamb = 1e-2 #suggested by HW; also try with my result of 1e-7
		alphas = [5e-3, 5e-2, 1e-1, "1/t", "1/sqrt(t)"]
		alphas = [5e-3, 5e-2, 1e-1, "1/t"]
		alphas = [5e-3, "1/t", "1/sqrt(t)"]

		theta_star = []
		loss_star = np.zeros(len(alphas))
		for i in xrange(len(alphas)):
			alpha = alphas[i]
			theta_hist, loss_hist, loss_by_step = stochastic_grad_descent(X, y, alpha=alpha, lambda_reg=lamb, num_iter=10)
			theta_star.append(theta_hist[-1][-1])
			loss_star[i] = loss_hist[-1]
			plt.plot(loss_hist, label="alpha="+str(alpha))
		plt.legend()
		#plt.yscale('log')
		#plt.xscale('log')
		plt.show()

if __name__ == "__main__":
    main()
