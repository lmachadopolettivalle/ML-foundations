import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from setup_problem import load_problem
from ridge_regression import RidgeRegression
import time
from sys import argv

class LassoRegression():
	def __init__(self, l1reg=0):
		if l1reg < 0:
			raise ValueError("L1 Regularization Penalty should be at least 0.")
		self.l1reg = l1reg
	
	def lasso_obj(self, X, y, w):
		n, num_features = X.shape
		predictions = np.dot(X, w)
		residual = predictions - y
		empirical_risk = np.sum(residual**2) / n
		l1_norm = np.linalg.norm(w, 1)
		objective = empirical_risk + l1_norm
		
		return objective
		
	def shooting_alg(self, X, y, start="RR", order="cyclic", num_iter=1000, epsilon=1e-8):
		"""
		Args:
		X - design matrix
		y - true outcomes
		start - initial guess for shooting algorithm
			"0" starts w = 0
			"RR" starts with ridge regression solution
		order - "cyclic" vs. "random"
		num_iter - number of iterations before stopping
		epsilon - convergence criterion; stop if old_obj - new_obj < epsilon
		"""
		n, num_features = X.shape
		if type(start) is not str:
			w = start
		else:
			if start == "0":
				w = np.zeros(num_features)
			else: # start == "RR"
				w = np.dot(np.linalg.inv(np.dot(np.transpose(X), X) + self.l1reg * np.eye(num_features)), np.dot(np.transpose(X), y))
		
		indices = np.arange(num_features)
		self.obj = self.lasso_obj(X, y, w)
		for i in xrange(num_iter):
			if order=="random":
				indices = np.random.permutation(num_features)
			for j in indices:
				a_j = 2 * np.dot(X[:, j], X[:, j])
				c_j = 2 * np.dot(X[:, j], y - np.dot(X, w) + w[j]*X[:, j])
				#c_j = 2 * np.dot(np.transpose(X), y - np.dot(X, w) + 0.5 * w[j] * a_j)[j]
				if a_j == 0:
					w[j] = 0
				elif c_j < -1*self.l1reg:
					w[j] = (c_j + self.l1reg) / a_j
				elif c_j > self.l1reg:
					w[j] = (c_j - self.l1reg) / a_j
				else:
					w[j] = 0

			old_obj = self.obj
			self.obj = self.lasso_obj(X, y, w)
			if np.absolute(old_obj - self.obj) < epsilon:
				break

		self.w = w
		return self
	
	def predict(self, X):
		try:
			getattr(self, "w")
		except AttributeError:
			raise RuntimeError("You must train before predicting!")
		return np.dot(X, self.w)
	
	def score(self, X, y):
		try:
			getattr(self, "w")
		except AttributeError:
			raise RuntimeError("You must train before predicting!")
		
		residuals = self.predict(X) - y
		return np.dot(residuals, residuals) / len(y) 



#solution to 2.1
def run_2_1(X_train, y_train, X_val, y_val, l2reg_search, print_table=True, PLOT=True):
	num_features = X_train.shape[1]	

	scores = np.zeros(len(l2reg_search))

	for i, l2reg in enumerate(l2reg_search):
		ridge_regression = RidgeRegression(l2reg=l2reg)
		ridge_regression.fit(X_train, y_train)
		scores[i] = ridge_regression.score(X_val, y_val)

	if PLOT:
		fig, ax = plt.subplots()
		ax.semilogx(l2reg_search, scores)
		ax.grid()
		ax.set_title("Validation Performance vs. L2 Regularization Parameter")
		ax.set_xlabel("L2-Penalty Regularization Parameter")
		ax.set_ylabel("Average Square Error")
		plt.show()
	
	#print vertical table of (l2reg, score)
	#TODO later, figure out cleaner way with pandas
	if print_table:
		print "L2_Parameter | Average Square Error"
		for i in xrange(len(l2reg_search)):
			print l2reg_search[i], "|", scores[i]


	#choose L2-parameter that minimizes score
	l2reg_opt = l2reg_search[np.argmin(scores)]

	return l2reg_opt

# solution to 2.2
# plot contains one or both of "PRED", "COEF"
def	run_2_2(x, x_train, y_train, pred_fns, plot=["PRED", "COEF"]): 
	if "PRED" in plot:
		fig, ax = plt.subplots()
		ax.scatter(x_train, y_train, label="Training Data")
		for i in xrange(len(pred_fns)):
			ax.plot(x, pred_fns[i]["preds"], label=pred_fns[i]["name"])

		ax.set_xlabel("Input Space: [0, 1)")
		ax.set_ylabel("Outcome Space")
		ax.set_title("Prediction Functions")
		ax.legend(loc="best")
		plt.show()
	
	if "COEF" in plot:
		fig, ax = plt.subplots(len(pred_fns), sharex=True)
		for i in xrange(len(pred_fns)):
			ax[i].plot(pred_fns[i]["coefs"], label=pred_fns[i]["name"])
			ax[i].legend(loc="best")

		ax[-1].set_xlabel("Feature Number")
		ax[len(pred_fns)/2].set_ylabel("Feature Coefficients / Weights")
		ax[0].set_title("Parameter Weight Distributions")
		plt.show()

# solution to 2.3
def run_2_3(coefs_true, coefs_opt, epsilon=1e-6):
	predicted = [1 if i >= epsilon else 0 for i in coefs_opt]
	true = [1 if i > 0 else 0 for i in coefs_true]
	
	confusion = confusion_matrix(true, predicted)
	TN, FP, FN, TP = confusion.ravel()
	print "Epsilon:", epsilon
	print "TN =", TN, "| FP =", FP, "| FN =", FN, "| TP =", TP

# solution to 3.2
def run_3_2(X_train, y_train, X_val, y_val, l1reg, epsilons=[1e-8]):
	begin = time.time()
	lasso_regression = LassoRegression(l1reg=l1reg)
	end = time.time()
	print "Doing regression with L1-reg =", l1reg, "takes", end-begin, "seconds"

	
	for start in ["0", "RR"]:
		for order in ["cyclic", "random"]:
			for epsilon in epsilons:
				begin = time.time()
				lasso_regression.shooting_alg(X_train, y_train, start=start, order=order, epsilon=epsilon)
				end = time.time()
				score = lasso_regression.score(X_val, y_val)
				print "0vsRR\tOrder\tEpsilon\tScore\tTime"
				print start, order, epsilon, score, end-begin, "seconds"

#solution to 3.3 a
def run_3_3_a(X_train, y_train, X_val, y_val, l1reg_search, print_table=True, PLOT=True):
	num_features = X_train.shape[1]	

	scores = np.zeros(len(l1reg_search))

	for i, l1reg in enumerate(l1reg_search):
		lasso_regression = LassoRegression(l1reg=l1reg)
		lasso_regression.shooting_alg(X_train, y_train)
		scores[i] = lasso_regression.score(X_val, y_val)

	if PLOT:
		fig, ax = plt.subplots()
		ax.semilogx(l1reg_search, scores)
		ax.grid()
		ax.set_title("Validation Performance vs. L1 Regularization Parameter")
		ax.set_xlabel("L1-Penalty Regularization Parameter")
		ax.set_ylabel("Average Square Error")
		plt.show()
	
	#print vertical table of (l1reg, score)
	#TODO later, figure out cleaner way with pandas
	if print_table:
		print "L1_Reg\t|\tAverage Square Error"
		for i in xrange(len(l1reg_search)):
			print l1reg_search[i], "\t|\t", scores[i]


	#choose L1-parameter that minimizes score
	l1reg_opt = l1reg_search[np.argmin(scores)]

	return l1reg_opt

# solution to 3.3 b
def	run_3_3_b(x, x_train, y_train, pred_fns, plot=["PRED", "COEF"]): 
	if "PRED" in plot:
		fig, ax = plt.subplots()
		ax.scatter(x_train, y_train, label="Training Data")
		for i in xrange(len(pred_fns)):
			ax.plot(x, pred_fns[i]["preds"], label=pred_fns[i]["name"])

		ax.set_xlabel("Input Space: [0, 1)")
		ax.set_ylabel("Outcome Space")
		ax.set_title("Prediction Functions")
		ax.legend(loc="best")
		plt.show()
	
	if "COEF" in plot:
		fig, ax = plt.subplots(len(pred_fns), sharex=True)
		for i in xrange(len(pred_fns)):
			ax[i].plot(pred_fns[i]["coefs"], label=pred_fns[i]["name"])
			ax[i].legend(loc="best")

		ax[-1].set_xlabel("Feature Number")
		ax[len(pred_fns)/2].set_ylabel("Feature Coefficients / Weights")
		ax[0].set_title("Parameter Weight Distributions")
		plt.show()

# solution to 3.4
def run_3_4(X_train, y_train, X_val, y_val, p=0.8, length=10):
	n, num_features = X_train.shape
	l_max = 2 * np.linalg.norm(np.dot(np.transpose(X_train), y_train), np.inf)

	l1reg = l_max * p**np.arange(length)
	scores = np.zeros(len(l1reg))
	
	start = np.zeros(num_features)
	for i in xrange(len(l1reg)):
		lasso = LassoRegression(l1reg=l1reg[i])
		lasso.shooting_alg(X_train, y_train, start=start)
		start = lasso.w
		scores[i] = lasso.score(X_val, y_val)

	fig, ax = plt.subplots()
	ax.scatter(l1reg, scores)
	ax.set_xscale('log')
	ax.set_xlabel("L1 Regularization Parameter = L_max * 0.8^i")
	ax.set_ylabel("Average Validation Square Loss")
	ax.set_title("Homotopy Method with Warm Starting")
	plt.show()


def main():
	if len(argv) == 2:
		program, sol = argv
	else:
		raise RuntimeError("USAGE: python solution.py OPTION[2 or 3]")

	# load data and split data
	lasso_data_fname = "lasso_data.pickle"
	x_train, y_train, x_val, y_val, target_fn, coefs_true, featurize = load_problem(lasso_data_fname)

	# turn from 1D binary data to high dimensional featurized data
	X_train = featurize(x_train)
	X_val = featurize(x_val)
	
	if sol == "2":
		#### 2.1
		# create array of possible L2Reg parameters
		l2reg_search = 10.**np.arange(-6, +1, 1)
		# search through l2reg_search
		l2reg_opt = run_2_1(X_train, y_train, X_val, y_val, l2reg_search, print_table=False, PLOT=False)
		
		#### 2.2
		# x has many inputs from 0 to 1, as well as the x_train inputs, to help plotting
		x = np.sort(np.concatenate([np.arange(0,1,.001), x_train]))
		X = featurize(x)
		
		# pred_fns is a list of dicts with "name", "coefs" and "preds"
		pred_fns = []
		coefs_opt = 0 #for question 2.3
		# first entry: Target function
		pred_fns.append({"name": "Target", "coefs": coefs_true, "preds": target_fn(x)})
		
		l2reg_values = [0, l2reg_opt]
		# next entries: prediction functions for L2Reg parameters in l2reg_values
		for l2reg in l2reg_values:
			ridge = RidgeRegression(l2reg=l2reg)
			ridge.fit(X_train, y_train)
			pred_fns.append({"name": "Ridge with L2Reg="+str(l2reg),
				"coefs": ridge.w_,
				"preds": ridge.predict(X)})
			# for question 2.3
			if l2reg == l2reg_opt:
				coefs_opt = ridge.w_
		# with pred_fns populated, plot
		# "PRED": prediction functions
		# "COEF": coefficients
		plots=["PRED", "COEF"]
		#plots=[]
		run_2_2(x, x_train, y_train, pred_fns, plot=plots)

		#### 2.3
		epsilon = []
		#epsilon = [1e-6, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1]
		for e in epsilon:
			run_2_3(coefs_true, coefs_opt, epsilon=e)

	if sol == "3":
		#### 3.2 - experiment with Lasso
		# Found that start="RR", order="cyclic", epsilon=1e-8 works MARGINALLY better
		#run_3_2(X_train, y_train, X_val, y_val, l1reg=1, epsilons=[1e-8, 1e-3])


		#### 3.3
		#### Part a: find optimal l1reg
		# create array of possible L1Reg parameters
		#l1reg_search = 10.**np.arange(-6, 2, 1)
		# search through l1reg_search
		#l1reg_opt = run_3_3_a(X_train, y_train, X_val, y_val, l1reg_search)
		l1reg_opt = 1.0 # found from above


		#### 3.3
		#### Part b: plot corresponding prediction function
		# x has many inputs from 0 to 1, as well as the x_train inputs, to help plotting
		x = np.sort(np.concatenate([np.arange(0,1,.001), x_train]))
		X = featurize(x)
		
		# pred_fns is a list of dicts with "name", "coefs" and "preds"
		pred_fns = []
		# first entry: Target function
		pred_fns.append({"name": "Target", "coefs": coefs_true, "preds": target_fn(x)})

		lasso = LassoRegression(l1reg=l1reg_opt)
		lasso.shooting_alg(X_train, y_train)

		pred_fns.append({"name": "Ridge with L1Reg="+str(l1reg_opt),
			"coefs": lasso.w,
			"preds": lasso.predict(X)})
		 
		# with pred_fns populated, plot
		# "PRED": prediction functions
		# "COEF": coefficients
		run_3_3_b(x, x_train, y_train, pred_fns, plot=[])

		run_3_4(X_train, y_train, X_val, y_val, p=0.8)

if __name__ == "__main__":
	main()
