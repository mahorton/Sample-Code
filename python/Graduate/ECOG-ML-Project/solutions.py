import numpy as np
import utils
import cPickle
from sklearn.preprocessing import scale, StandardScaler



def gb_full():
	from sklearn.ensemble import GradientBoostingRegressor
	X_train = np.loadtxt("train_X_ecog.csv", delimiter=",")
	X_test = np.loadtxt("test_X_ecog.csv", delimiter=",")
	y = np.loadtxt("train_Y_ecog.csv", delimiter=",")
	y_mean = y.mean(axis=0)
	y -= y_mean
	#X_train, X_test = utils.transform_data(X_train, X_test)
	inds = utils.get_important_features(12)
	
	X_train = X_train[:,inds]
	X_test = X_test[:,inds]
	
	stacks = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
	for i in stacks:
		X_train = np.hstack((X_train, np.roll(X_train[:,:12], -i, axis=0)))
		X_test = np.hstack((X_test, np.roll(X_test[:,:12], -i, axis=0)))

	X_train = scale(X_train)
	X_test = scale(X_test)
	
	y_pred = np.zeros((X_test.shape[0], y.shape[1]))
	train_pred = np.zeros_like(y)
	for i in xrange(y_pred.shape[1]):
		regr = GradientBoostingRegressor(learning_rate=.07, n_estimators=40, max_depth=4)
		#regr = MultiOutputRegressor(regr0, 2)
		regr.fit(X_train, y[:,i])
		train_pred[:,i] = regr.predict(X_train)
		y_pred[:,i] = regr.predict(X_test)

	mse = utils.calc_rmse(train_pred, y)
	print mse
	y_pred += y_mean

	
	utils.generateSubmission(y_pred, "gb3.csv")



def ridgeregr():
	from sklearn.decomposition import PCA
	from sklearn.linear_model import Ridge

	X_train = np.loadtxt("train_X_ecog.csv", delimiter=",")
	X_test = np.loadtxt("test_X_ecog.csv", delimiter=",")
	y = np.loadtxt("train_Y_ecog.csv", delimiter=",")
	y_mean = y.mean(axis=0)
	y -= y_mean
	#X_train, X_test = utils.transform_data(X_train, X_test)
	X_train = scale(X_train)
	X_test = scale(X_test)

	inds = utils.get_important_features(10)
	X_train = X_train[:,inds]
	X_test = X_test[:,inds]

	regr = Ridge(alpha=3e5)
	regr.fit(X_train, y)

	train_pred = regr.predict(X_train)

	mse = utils.calc_rmse(train_pred, y)
	print mse

	y_pred = regr.predict(X_test)
	y_pred += y_mean

	utils.generateSubmission(y_pred, "ridge1.csv")


def gb():
	from sklearn.ensemble import GradientBoostingRegressor
	from sklearn.decomposition import PCA
	X_train = np.loadtxt("train_X_ecog.csv", delimiter=",")
	X_test = np.loadtxt("test_X_ecog.csv", delimiter=",")
	y = np.loadtxt("train_Y_ecog.csv", delimiter=",")
	y0 = y.copy()
	y_pca = PCA(n_components=1)
	y_pca.fit(y)
	y = y_pca.transform(y)

	#X_train, X_test = utils.transform_data(X_train, X_test)
	inds = utils.get_important_features(12)
	
	X_train = X_train[:,inds]
	X_test = X_test[:,inds]
	
	stacks = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
	for i in stacks:
		X_train = np.hstack((X_train, np.roll(X_train[:,:12], -i, axis=0)))
		X_test = np.hstack((X_test, np.roll(X_test[:,:12], -i, axis=0)))

	X_train = scale(X_train)
	X_test = scale(X_test)
	

	regr = GradientBoostingRegressor(learning_rate=.07, n_estimators=40, max_depth=4)
	#regr = MultiOutputRegressor(regr0, 2)
	regr.fit(X_train, y)

	train_pred = regr.predict(X_train)
	train_pred.shape = ((train_pred.shape[0],1))
	train_pred = y_pca.inverse_transform(train_pred)
	
	mse = utils.calc_rmse(train_pred, y0)
	print mse

	y_pred = regr.predict(X_test)
	y_pred.shape = ((y_pred.shape[0],1))
	y_pred = y_pca.inverse_transform(y_pred)

	utils.generateSubmission(y_pred, "gb3.csv")


def neuralnet():
	from sklearn.neural_network import MLPRegressor
	from sklearn.decomposition import PCA
	X_train = np.loadtxt("train_X_ecog.csv", delimiter=",")
	X_test = np.loadtxt("test_X_ecog.csv", delimiter=",")
	y0 = np.loadtxt("train_Y_ecog.csv", delimiter=",")
	y = y0.copy()
	y_pca = PCA(n_components=2)
	y_pca.fit(y)
	y = y_pca.transform(y)
	y = utils.roll(y, n_rolls=16)

	y_mean = y.mean(axis=0)
	y -= y_mean

	inds = utils.get_important_features(12)
	X_train = X_train[:,inds]
	X_test = X_test[:,inds]
	
	stacks = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
	for i in stacks:
		X_train = np.hstack((X_train, np.roll(X_train[:,:12], -i, axis=0)))
		X_test = np.hstack((X_test, np.roll(X_test[:,:12], -i, axis=0)))

	X_train = scale(X_train)
	X_test = scale(X_test)
	#y = utils.roll_y(y)

	#regr = MLPRegressor(hidden_layer_sizes=(1550,1500), alpha=10000., learning_rate="adaptive",max_iter=10000)
	regr = MLPRegressor(hidden_layer_sizes=(350,556), activation="logistic", alpha=35., learning_rate="adaptive",max_iter=10000)
	regr.fit(X_train, y)
	#print utils.calc_rmse(regr.predict(X), y)
	train_pred = regr.predict(X_train)

	train_pred += y_mean
	train_pred = utils.unroll(train_pred, n_rolls=16)
	train_pred = y_pca.inverse_transform(train_pred)

	train_mse = utils.calc_rmse(train_pred, y0)
	print train_mse
	#X = scale(X)
	y_pred = regr.predict(X_test) 

	y_pred += y_mean
	y_pred = utils.unroll(y_pred, n_rolls=16)
	y_pred = y_pca.inverse_transform(y_pred)
	
	#y_pred = utils.unroll_y(y_pred) + y_mean
	utils.generateSubmission(y_pred, "nnr3.csv")



def randomforest():
	from sklearn.ensemble import RandomForestRegressor
	X, y = utils.load_data(source="rf_pca_X.csv", scaled=False, load_test=False)

	regr = RandomForestRegressor(n_estimators=450, max_features=300)
	regr.fit(X, y)
	#print max(regr.feature_importances_.tolist()), regr.feature_importances_.tolist().index(max(regr.feature_importances_.tolist()))
	#print min(regr.feature_importances_.tolist()), regr.feature_importances_.tolist().index(min(regr.feature_importances_.tolist())) 
	train_pred = regr.predict(X)
	mse = utils.calc_mse(train_pred, y)
	print mse

	#X_test = cPickle.load(open("saves\\X_test_stacked_pca.p", "rb"))
	X = np.loadtxt("saves\\test_rf_pca_X.csv", delimiter=",")
	#X_test = np.loadtxt("test_X_ecog.csv", delimiter=",")
	y_pred = regr.predict(X)
	
	utils.generateSubmission(y_pred, "rf4.csv")



def extratrees():
	from sklearn.ensemble import ExtraTreesRegressor

	#X = cPickle.load(open("saves\\X_stacked_pca.p", "rb"))
	X = np.loadtxt("saves\\stacked_X.csv", delimiter=",")
	y = np.loadtxt("train_Y_ecog.csv", delimiter=",")

	regr = ExtraTreesRegressor(n_estimators=300, max_features=100, max_depth=35, min_samples_leaf=20, bootstrap=True, oob_score=True)
	
	y = utils.roll_y(y)
	regr.fit(X, y)
	#print max(regr.feature_importances_.tolist()), regr.feature_importances_.tolist().index(max(regr.feature_importances_.tolist()))
	#print min(regr.feature_importances_.tolist()), regr.feature_importances_.tolist().index(min(regr.feature_importances_.tolist())) 
	train_pred = regr.predict(X)
	train_pred = utils.unroll_y(train_pred)
	y = utils.unroll_y(y)
	mse = utils.calc_rmse(train_pred, y)
	print mse

	#X_test = cPickle.load(open("saves\\X_test_stacked_pca.p", "rb"))
	X_test = np.loadtxt("saves\\test_stacked_X.csv", delimiter=",")
	y_pred = regr.predict(X_test)
	y_pred = utils.unroll_y(y_pred)

	utils.generateSubmission(y_pred, "et3.csv")





