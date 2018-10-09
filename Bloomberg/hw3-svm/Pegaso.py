from collections import Counter
from util import *

class Pegaso():
	def __init__(self, l2reg=1):
		if l2reg <= 0:
			raise ValueError("L2 Reg Parameter must be non-zero and positive!")
		self.l2reg = l2reg
	
	def alg1(self, data, epochs=3):
		epoch = 0
		t = 0
		m = len(data)

		w = {}
		# termination: number of epochs
		while epoch < epochs:
			print epoch
			#w_old = w # for termination based on convergence
			for j in xrange(m):
				#print j
				t += 1
				step_j = 1.0 / (t*self.l2reg)
				y_j, bag_j = make_bag(data[j])
				
				increment(w, (-1 * step_j * self.l2reg), w)
				if y_j * dotProduct(w, bag_j) < 1:
					increment(w, step_j * y_j, bag_j)
			epoch += 1

		return w

	def alg2(self, train_data, epochs=100, epsilon=1e-8):
		epoch = 0
		t = 0
		m = len(train_data)

		s = 1
		W = Counter()
		old_loss = 1
		# termination: number of epochs
		while epoch < epochs:
			print epoch
			for j in xrange(m):
				#print j
				t += 1
				step_j = 1.0 / (t*self.l2reg)
				y_j, bag_j = make_bag(train_data[j])
				
				s *= (1 - step_j * self.l2reg)
				if s == 0:
					s = 1
					W = Counter()
					continue
				if y_j * s * dotProduct(W, bag_j) < 1:
					increment(W, step_j * y_j / s, bag_j)
			epoch += 1
			
			# for termination based on convergence
			w = Counter()
			increment(w, s, W)
			new_loss = self.loss(w, train_data)
			if old_loss - new_loss < epsilon:
				break
			old_loss = new_loss
		
		return w

	def loss(self, w, val_data):
		m = len(val_data)
		loss = 0.0
		def sign(x):
			if x >= 0:
				return 1
			else:
				return -1

		for j in xrange(m):
			y_j, bag_j = make_bag(val_data[j])
			if sign(y_j) != sign(dotProduct(w, bag_j)):
				loss += 1

		loss /= m
		return loss


