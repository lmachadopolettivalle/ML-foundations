from __future__ import division

import os
import numpy as np
import pickle
import random
import time
import matplotlib.pyplot as plt
from string import maketrans # for Python 2
from collections import Counter
from util import *
import operator # to print sorted dict on 7.1


'''
Note:  This code is just a hint for people who are not familiar with text processing in python. There is no obligation to use this code, though you may if you like. 
'''


def folder_list(path,label):
    '''
    PARAMETER PATH IS THE PATH OF YOUR LOCAL FOLDER
    '''
    filelist = os.listdir(path)
    review = []
    for infile in filelist:
        file = os.path.join(path,infile)
        r = read_data(file)
        r.append(label)
        review.append(r)
    return review

def read_data(file):
    '''
    Read each file into a list of strings. 
    Example:
    ["it's", 'a', 'curious', 'thing', "i've", 'found', 'that', 'when', 'willis', 'is', 'not', 'called', 'on', 
    ...'to', 'carry', 'the', 'whole', 'movie', "he's", 'much', 'better', 'and', 'so', 'is', 'the', 'movie']
    '''
    f = open(file)
    lines = f.read().split(' ')
    symbols = '_!?${}()[].,:;+-*/&|<>=~" '
    #words = map(lambda Element: Element.translate(str.maketrans("", "", symbols)).strip(), lines) #only works in Python 3
    words = map(lambda Element: Element.translate(maketrans(symbols, " "*len(symbols))).strip(), lines)
    words = filter(None, words)
    return words
	
###############################################
######## YOUR CODE STARTS FROM HERE. ##########
###############################################

def shuffle_and_split_data(num=1500, rewrite=False):
	'''
	pos_path is where you save positive review data.
	neg_path is where you save negative review data.
	'''
	try:
		review = pickle.load(open("data/Data.p", "rb"))
		if rewrite:
			random.shuffle(review)
			os.remove("data/Data.p")
			pickle.dump(review, open("data/Data.p", "wb"))
	except:
		pos_path = "data/pos"
		neg_path = "data/neg"

		pos_review = folder_list(pos_path,1)
		neg_review = folder_list(neg_path,-1)

		review = pos_review + neg_review
		random.shuffle(review)

		pickle.dump(review, open("data/Data.p", "wb"))
	finally:
		return review[:num], review[num:]



'''
Now you have read all the files into list 'review' and it has been shuffled.
Save your shuffled result by pickle.
*Pickle is a useful module to serialize a python object structure. 
*Check it out. https://wiki.python.org/moin/UsingPickle
'''
 
from Pegaso import Pegaso

#Time for Algo. 1: 85.8879520893 seconds
#Loss for Algo. 1: 0.364
#Time for Algo. 2: 3.34364414215 seconds
#Loss for Algo. 2: 0.374
def run_6_6(train_data, val_data, l2reg=1, epochs=2):
	pegaso = Pegaso(l2reg=l2reg)
	
	# run algorithm 1, record time and loss
	begin1 = time.time()
	w1 = pegaso.alg1(train_data, epochs=epochs)
	end1 = time.time()
	loss1 = pegaso.loss(w1, val_data)
	print "Time for Algo. 1:", end1-begin1, "seconds"
	print "Loss for Algo. 1:", loss1

	# run algorithm 2, record time and loss
	begin2 = time.time()
	w2 = pegaso.alg2(train_data, epochs=epochs)
	end2 = time.time()
	loss2 = pegaso.loss(w2, val_data)
	print "Time for Algo. 2:", end2-begin2, "seconds"
	print "Loss for Algo. 2:", loss2


#Prints array of l2reg parameters and array of corresponding losses
#Plots "L2 Parameter Search.pdf"
def run_6_8(train_data, val_data, l2reg_search, epochs=100, plot=True):
	losses = np.zeros(len(l2reg_search))
	for i in xrange(len(l2reg_search)):
		pegaso = Pegaso(l2reg=l2reg_search[i])
		w = pegaso.alg2(train_data, epochs=epochs)
		losses[i] = pegaso.loss(w, val_data)
	
	print "L2_Search:", l2reg_search
	print "Losses:", losses

	fig, ax = plt.subplots()
	ax.semilogx(l2reg_search, losses)
	ax.set_xlabel("L2 Regularization Parameter")
	ax.set_ylabel("Average Validation 0-1 Loss")
	ax.set_title("L2 Regularization Hyperparameter Tuning")
	plt.savefig("./figures/Recent L2 Parameter Search.pdf")
	plt.show()

def run_6_9(train_data, val_data, l2reg):
	pegaso = Pegaso(l2reg=l2reg)
	w = pegaso.alg2(train_data)

	scores = [dotProduct(w, make_bag(i)[1]) for i in val_data]
	predictions = [1 if scores[i] >= 0 else -1 for i in xrange(len(scores))]
	corrects = [make_bag(i)[0] for i in val_data]
	margins = [c*s for (c,s) in zip(corrects, scores)]
	
	#sort scores and reorder corrects based on sorting of scores
	mag_scores = [s if s >= 0 else -1*s for s in scores]
	sorted_tuple = [(s, c, p) for s, c, p in sorted(zip(mag_scores, corrects, predictions), key=lambda pair: pair[0])]


	def chunks(l, n):
		"""Yield successive n-sized chunks from l."""
		for i in range(0, len(l), n):
			#yield l[i:i + n]
			yield np.arange(i, min(i+n, len(l)))

	L = len(scores)
	n = 10

	groups = chunks(margins, int(L / n))


	print "Group#  GroupSize  PercentageError  Min |score| in Group  Max |score| in Group"
	print "-"*70
	counter = 0

	percentage_error_array = []
	min_score_array = []

	for group in groups:
		counter += 1
		size = 0
		error = 0
		get_min_score = True
		for i in group:
			max_score = sorted_tuple[i][0]

			if get_min_score:
				min_score = sorted_tuple[i][0]
				get_min_score = False
			size += 1
			if sorted_tuple[i][1] != sorted_tuple[i][2]:
				error += 1

		if size != 0:
			error /= size
		else:
			error = "NA"

		print counter, size, error, min_score, max_score
		percentage_error_array.append(error)
		min_score_array.append(min_score)
	
	plt.plot(min_score_array, percentage_error_array)
	plt.xlabel("Min. absolute score in each group", fontsize=14)
	plt.ylabel("Percentage Error in the group", fontsize=14)
	plt.title("Percentage error in each of {0} groups separated by absolute score", fontsize=13)
	plt.savefig("./figures/Error vs Score.pdf")
	plt.show()


def run_7_1(train_data, val_data, l2reg):
	pegaso = Pegaso(l2reg=l2reg)
	w = pegaso.alg2(train_data)
	scores = [dotProduct(w, make_bag(i)[1]) for i in val_data]
	corrects = [make_bag(i)[0] for i in val_data]
	margins = np.array([scores[i]*corrects[i] for i in xrange(len(scores))])

	wrong_indices = np.where(margins < 0)[0]
	i = np.random.choice(wrong_indices)
	y, x = make_bag(val_data[i])

	w_x = {i: abs(w[i]*x[i]) for i in x}

	sorted_w_x = sorted(w_x.items(), key=operator.itemgetter(1))
	print sorted_w_x


def main():
	train_data, val_data = shuffle_and_split_data(num=1500, rewrite=True)

	#### 6.6
	# Compare times for two different Pegaso implementations
	# Immediately conclude Algo. 2 is better
	#run_6_6(train_data, val_data, l2reg=3)
	
	#### 6.8
	# Use Algo. 2 to do a parameter search for optimal lambda
	# Found lambda = 0.05 is close enough to optimal
	#l2reg_search = 10.**np.arange(-2, 1, 0.1) #wider range
	#l2reg_search = 10.**np.arange(-1, 0.5, 0.1) #narrow range
	#run_6_8(train_data, val_data, l2reg_search)
	l2_opt = 0.05

	#### 6.9
	# Determine whether the magnitude of the score
	# correlates with the confidence of our prediction
	#run_6_9(train_data, val_data, l2_opt)

	#### 7.1
	# Study incorrect case and weights of features
	#run_7_1(train_data, val_data, l2_opt)



if __name__ == "__main__":
	main()
