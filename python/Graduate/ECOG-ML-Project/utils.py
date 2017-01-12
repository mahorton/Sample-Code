import numpy as np
import csv
import cPickle
import copy
#import threading
#import Queue
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.kernel_approximation import RBFSampler



def scale_data(X_train, X_test):

	scaler = StandardScaler()
	scaler.fit(X_train)
	return scaler.transform(X_train), scaler.transform(X_test)


def split_data(X, y, ratio=.7, group_size=200, seed=12345):

	n,p = X.shape
	inds = range(n)
	n_groups = n/group_size
	groups = []
	for i in xrange(n_groups):
		groups.append(inds[group_size*i:group_size*(i+1)])
	groups.append(inds[group_size*n_groups:])
	np.random.seed(seed)
	np.random.shuffle(groups)
	inds = []
	for g in groups:
		inds += g

	train_inds = inds[:int(ratio*n)]
	val_inds = inds[int(ratio*n):]
	if len(inds) != n:
		print "left out some observations", len(inds),"/",n
	return train_inds, val_inds


def transform_data3(X_train, X_test, n_joins=5):

	temp_X_train = np.zeros((X_train.shape[0], X_train.shape[1], n_joins))
	temp_X_test = np.zeros((X_test.shape[0], X_test.shape[1], n_joins))

	for i in xrange(n_joins):
		temp_X_train[:,:,i] = np.roll(X_train, -i, axis=0)
		temp_X_test[:,:,i] = np.roll(X_test, -i, axis=0)

	new_X_train = np.zeros_like(X_train)
	new_X_test = np.zeros_like(X_test)

	for i in xrange(X_train.shape[1]):
		pca = PCA(n_components=1)
		pca.fit(temp_X_train[:,i,:])
		new_X_train[:,i] = pca.transform(temp_X_train[:,i,:])[:,0]
		new_X_test[:,i] = pca.transform(temp_X_test[:,i,:])[:,0]
		#print pca.explained_variance_ratio_

	scaler = StandardScaler()
	scaler.fit(new_X_train)
	X_train = scaler.transform(new_X_train)
	X_test = scaler.transform(new_X_test)
	#add squared and cubed terms
	"""
	new_X_train = np.hstack((new_X_train, new_X_train**2, new_X_train**3))
	new_X_test = np.hstack((new_X_test, new_X_test**2, new_X_test**3))
	"""

	return new_X_train, new_X_test


def transform_data2(X_train, X_test):

	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)

	X_train.shape = (X_train.shape[0], 6, 70)
	X_test.shape = (X_test.shape[0], 6, 70)
	new_X_train = np.zeros((X_train.shape[0], 176))
	new_X_test = np.zeros((X_test.shape[0], 176))

	for i in xrange(70):
		pca = PCA(n_components=2)
		pca.fit(X_train[:,:,i])
		new_X_train[:,2*i:2*(i+1)] = pca.transform(X_train[:,:,i])
		new_X_test[:,2*i:2*(i+1)] = pca.transform(X_test[:,:,i])
		#print pca.explained_variance_ratio_

	for i in xrange(6):
		pca = PCA(n_components=6)
		pca.fit(X_train[:,i,:])
		new_X_train[:,140+6*i:140+6*(i+1)] = pca.transform(X_train[:,i,:])
		new_X_test[:,140+6*i:140+6*(i+1)] = pca.transform(X_test[:,i,:])
		#print pca.explained_variance_ratio_

	new_X_train = np.hstack((new_X_train, np.roll(new_X_train, -1, axis=0)))
	new_X_test = np.hstack((new_X_test, np.roll(new_X_test, -1, axis=0)))

	scaler = StandardScaler()
	scaler.fit(new_X_train)
	new_X_train = scaler.transform(new_X_train)
	new_X_test = scaler.transform(new_X_test)
	#add squared and cubed terms
	"""
	new_X_train = np.hstack((new_X_train, new_X_train**2, new_X_train**3))
	new_X_test = np.hstack((new_X_test, new_X_test**2, new_X_test**3))
	"""

	return new_X_train, new_X_test

def transform_data(X_train, X_test):

	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)

	X_train.shape = (X_train.shape[0], 6, 70)
	X_test.shape = (X_test.shape[0], 6, 70)
	new_X_train = np.zeros((X_train.shape[0], 88))
	new_X_test = np.zeros((X_test.shape[0], 88))

	for i in xrange(70):
		pca = PCA(n_components=1)
		pca.fit(X_train[:,:,i])
		new_X_train[:,i] = pca.transform(X_train[:,:,i])[:,0]
		new_X_test[:,i] = pca.transform(X_test[:,:,i])[:,0]
		#print pca.explained_variance_ratio_

	for i in xrange(6):
		pca = PCA(n_components=3)
		pca.fit(X_train[:,i,:])
		new_X_train[:,70+3*i:70+3*(i+1)] = pca.transform(X_train[:,i,:])
		new_X_test[:,70+3*i:70+3*(i+1)] = pca.transform(X_test[:,i,:])
		#print pca.explained_variance_ratio_

	new_X_train = np.hstack((new_X_train, np.roll(new_X_train, -2, axis=0)))
	new_X_test = np.hstack((new_X_test, np.roll(new_X_test, -2, axis=0)))

	#add squared and cubed terms
	"""
	new_X_train = np.hstack((new_X_train, new_X_train**2, new_X_train**3))
	new_X_test = np.hstack((new_X_test, new_X_test**2, new_X_test**3))
	"""

	return new_X_train, new_X_test

#2,12,2,2,13,15,3,2,11,11,12,54,21
def build_mean_data(n_stacks=6):
	X = np.loadtxt("train_X_ecog.csv", delimiter=",")
	y = np.loadtxt("train_Y_ecog.csv", delimiter=",")
	X.shape = (X.shape[0], 6, 70)
	X = X.mean(axis=1)
	#X = np.min(X, axis=2)
	X0 = X.copy()
	for i in xrange(1,n_stacks):
		X = np.hstack((X, np.roll(X0, -i, axis=0)))
	points = np.loadtxt("train_breakpoints.txt")
	inds = np.zeros(len(points)*(n_stacks-1))
	for i in xrange(1,n_stacks):
		inds[(i-1)*len(points):i*len(points)] = points + i
	np.savetxt("saves\\stacked_mean_X.csv", X, delimiter=",")

	X = np.loadtxt("test_X_ecog.csv", delimiter=",")
	X.shape = (X.shape[0], 6, 70)
	X = X.mean(axis=1)
	#X = np.min(X, axis=2)
	X0 = X.copy()
	for i in xrange(1,n_stacks):
		X = np.hstack((X, np.roll(X0, -i, axis=0)))
	
	np.savetxt("saves\\stacked_mean_X.csv", X, delimiter=",")

def update_bests(file, key, val):

	d = cPickle.load(open("saves\\"+file+".p", "rb"))
	d[key] = val
	cPickle.dump(d, open("saves\\"+file+".p", "wb"))

def rbf(X=None, gamma=1., save=True):

	if X is None:
		X = np.loadtxt("train_X_ecog.csv", delimiter=",")
		n,p = X.shape
		X = np.vstack((X, np.loadtxt("test_X_ecog.csv", delimiter=",")))
		X = scale(X)
		train_only=False
	else:
		train_only = True
		save = False

	sampler = RBFSampler(gamma=gamma, n_components=1200)
	X = sampler.fit_transform(X)

	if save:
		np.savetxt("saves\\X_rbfsample.csv", X[:n,:], delimiter=",")
		np.savetxt("saves\\test_X_rbfsample.csv", X[n:,:], delimiter=",")

	if train_only:
		return X
	else:
		return X[:n,:], X[n:,:]



def multithread_fitter(model, X, y, X_test=None,  n_threads=2, verbose=False):

	exitFlag = 0

	train_pred = np.zeros_like(y)
	y_pred = np.zeros((X_test.shape[0], y.shape[1]))
	if X_test is None:
		fitted_models = []

	class myThread (threading.Thread):
		def __init__(self, threadID, name, q):
			threading.Thread.__init__(self)
			self.threadID = threadID
			self.name = name
			self.q = q
		def run(self):
			if verbose: print "Starting " + self.name
			process_data(self.name, self.q)
			if verbose: print "Exiting " + self.name

	def process_data(threadName, q):
		while not exitFlag:
			queueLock.acquire()
			if not workQueue.empty():
				ind = q.get()
				regr = copy.copy(model)
				queueLock.release()
				if verbose: print "%s processing %s" % (threadName, ind)
				
				regr.fit(X, y[:,ind])
				train_pred[:,ind] = regr.predict(X)
				if X_test is None:
					fitted_models.append(regr)
				else:
					y_pred[:,ind] = regr.predict(X_test)
			else:
				queueLock.release()
			#time.sleep(1)

	threadList = ["Thread-1", "Thread-2", "Thread-3", "Thread-4"]
	threadList = threadList[:n_threads]
	nameList = range(32)
	queueLock = threading.Lock()
	workQueue = Queue.Queue(32)
	threads = []
	threadID = 1

	# Create new threads
	for tName in threadList:
		thread = myThread(threadID, tName, workQueue)
		thread.start()
		threads.append(thread)
		threadID += 1

	# Fill the queue
	queueLock.acquire()
	for word in nameList:
		workQueue.put(word)
	queueLock.release()

	# Wait for queue to empty
	while not workQueue.empty():
		pass

	# Notify threads it's time to exit
	exitFlag = 1

	# Wait for all threads to complete
	for t in threads:
		t.join()
	if X_test is None:
		return fitted_models, train_pred
	else:
		return y_pred, train_pred


def make_kernel_pca():

	X, y = load_data(load_test=False)

	pca = KernelPCA(kernel="rbf", gamma=.01, eigen_solver="arpack", remove_zero_eig=True)
	pca.fit(X)
	X = pca.transform(X)
	print X.shape
	np.savetxt("saves\\kpca_X.csv", X, delimiter=",")

	X = np.loadtxt("test_X_ecog.csv", delimiter=",")
	np.savetxt("saves\\test_kpca_X.csv", pca.transorm(X), delimiter=",")



def make_rolled_data():

	X_train = np.loadtxt("train_X_ecog.csv", delimiter=",")
	X_test = np.loadtxt("test_X_ecog.csv", delimiter=",")

	n = X_train.shape[0]

	#X_train = np.hstack((X_train, np.roll(X_train, -1, axis=0)))
	#X_test = np.hstack((X_test, np.roll(X_test, -1, axis=0)))

	X = np.vstack((np.hstack((X_train, np.roll(X_train, -1, axis=0))), np.hstack((X_test, np.roll(X_test, -1, axis=0)))))

	X = scale(X)

	np.savetxt("saves\\scaled_stacked_X.csv", X[:n,:], delimiter=",")
	np.savetxt("saves\\test_scaled_stacked_X.csv", X[n:,:], delimiter=",")



def calc_mse(y, y_hat):
	n,k = y.shape
	return np.sum((y - y_hat)**2)/(1.*n*k)

def calc_rmse(y, y_hat):
	if len(y.shape) == 2:
		n,k = y.shape
		return np.sqrt(np.sum((y - y_hat)**2)/(1.*n*k))
	else:
		return np.sqrt(np.sum((y - y_hat)**2)/(1.*len(y)))


def generateSubmission(y_pred, filename="submission.csv"):

	filename = "saves\\"+filename
	out = [["Id","Prediction"]]
	m,n = y_pred.shape
	y_pred = y_pred.T.reshape(m*n,)
	for i in xrange(1, m*n+1):
		out.append([i, y_pred[i-1]])

	with open(filename, "wb") as f:
		writer = csv.writer(f)
		writer.writerows(out)


def plot_ts(inds=range(420)[::6]):
	from matplotlib import pyplot as plt

	X_train = np.loadtxt("train_X_ecog.csv", delimiter=",")
	for i in inds:
		plt.plot(np.arange(X_train.shape[0]), X_train[:,i])
	plt.show()

def plot_y(interval=(0,-1)):
	from matplotlib import pyplot as plt
	y = np.loadtxt("train_Y_ecog.csv", delimiter=",")

	for i in xrange(32):
		plt.plot(y[:,i][interval[0]:interval[1]])
	plt.show()

def gen_new_h(tup, count, magnitude=100, expand_out=False):

	n = len(tup)
	new_h = []
	for i in xrange(n):

		copy = list(tup)
		copy[i] += magnitude/(count+1)
		new_h.append(tuple(copy))

		copy = list(tup)
		copy[i] = max(10, copy[i] - magnitude/(count+1))
		new_h.append(tuple(copy))

		if expand_out:
			copy = list(tup)
			copy = copy[:i] + [copy[i]] + copy[i:]
			new_h.append(tuple(copy))

	return new_h

def roll(y0, n_rolls=1):

	y = y0.copy()
	m,n = y0.shape
	for i in xrange(n_rolls):
		y = np.hstack((y, np.roll(y0, -i, axis=0)))

	return y

def unroll(y0, n_rolls=1):

	n,p = y0.shape
	r = p/n_rolls
	y = np.zeros((n, r))
	for i in xrange(n_rolls):
		y += np.roll(y0[:,i*r:(i+1)*r], i, axis=0)
	y /= n_rolls

	return y


def gen_next_lambda(mse, train_mse, prev_mse, prev_lam, step_size, epsilon=.001):

	if mse - train_mse > 3:
		return prev_lam + step_size*3
	elif mse - train_mse > 2:
		return prev_lam + step_size*2

	elif prev_mse + epsilon < mse or train_mse > 16:
		return False

	else:
		return prev_lam + step_size


def plot_filters():

	from matplotlib import pyplot as plt
	regr = cPickle.load(open("nnr.p", "rb"))
	Z = regr.coefs_[0]
	Z = Z[:250,:]

	m,n = Z.shape
	X,Y = np.meshgrid(np.arange(m), np.arange(n))
	Z2 = Z[X,Y]

	plt.pcolormesh(X,Y,Z2)
	plt.xlabel("Index")
	plt.ylabel("First layer weights")
	
	plt.show()



def get_important_features(n_return, plot=False):

	gb = np.loadtxt("saves\\gb_feature_importances.csv")
	lasso = np.loadtxt("saves\\lasso_coefs.csv")
	rf = np.loadtxt("saves\\rf_feature_importances.csv")
	#without rf
	#362,142,2,83,155,213,212,361,151,152,404,371
	gb = gb/np.sum(gb)
	lasso = lasso/np.sum(lasso)
	rf = rf/np.sum(rf)

	feats = (gb + lasso + rf)/3.
	feats = (gb + rf)/2.
	
	if plot:
		from matplotlib import pyplot as plt
		plt.plot(np.arange(len(feats)), feats)
		plt.xlabel("Feature Index")
		plt.ylabel("Importance")
		plt.title("Average Feature Importances")
		plt.show()

	#feats = lasso.copy()
	inds = []
	for i in xrange(n_return):
		ind = np.argmax(feats)
		inds.append(ind)
		feats[ind] = 0
	return inds