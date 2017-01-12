import numpy as np
import cPickle
import utils
#from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.preprocessing import StandardScaler, scale#, PolynomialFeatures
#from sklearn.kernel_ridge import KernelRidge
#from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_predict
#from sklearn.multioutput import MultiOutputRegressor


def nnr(max_repeat=3, expand_out=True, epsilon=.001):
	from sklearn.neural_network import MLPRegressor
	from sklearn.decomposition import PCA
	X = np.loadtxt("train_X_ecog.csv", delimiter=",")
	y = np.loadtxt("train_Y_ecog.csv", delimiter=",")
	y0 = y.copy()
	#avg_sentence = utils.get_avg_sentence()
	#y = utils.sub_avg_sentence(y, np.loadtxt("train_breakpoints.txt"), avg_sentence)
	n_rolls = 8

	tr_inds, val_inds = utils.split_data(X,y, group_size=200)
	X_train = X[tr_inds,:]
	y_train = y[tr_inds,:]
	X_val = X[val_inds,:]
	y_val = y[val_inds,:]
	
	y = y0.copy()
	y_pca = PCA(n_components=3)
	y_pca.fit(y)
	y = y_pca.transform(y)
	y = utils.roll(y, n_rolls=n_rolls)

	tr_inds, val_inds = utils.split_data(X,y, group_size=200)
	X_train = X[tr_inds,:]
	y_train = y[tr_inds,:]
	y_train0 = y0[tr_inds,:]
	X_val = X[val_inds,:]
	y_val = y0[val_inds,:]


	y_mean = y_train.mean(axis=0)
	y_train-=y_mean
	#y_mean = y_train.mean(axis=0)
	#y_train -= y_mean

	electrodes = list(set(np.array(utils.get_important_features(5))%70))
	inds = []
	for i in electrodes:
		inds += range(420)[i::70]

	print inds
	print len(inds)
	X_train = X_train[:,inds]
	X_val = X_val[:,inds]
	
	stacks = [-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
	for i in stacks:
		X_train = np.hstack((X_train, np.roll(X_train[:,:len(inds)], -i, axis=0)))
		X_val = np.hstack((X_val, np.roll(X_val[:,:len(inds)], -i, axis=0)))

	X_train = scale(X_train)
	X_val = scale(X_val)
	#bests = cPickle.load(open("saves\\bests5.p", "rb"))
	repeat = 0
	#reset bests
	bests = {"mse": 15.3, "h" : (250,), "lam" : 100}
	bests["mse"] = 15.4
	bests["activation"] = "logistic"
	
	hs = utils.gen_new_h(bests["h"],0,magnitude=50,expand_out=expand_out) + [(350,600),(155,156,157,158,160)]
	#hs = [(200,),(100,100,100),(50,50,100,100),(100,50,100,50),(50,100,50,100)]
	
	
	#hs = [(350,300)]
	#lams = [7000.,]
	while repeat < max_repeat:
		updated = False
		lam_step = 20./(repeat+1)

		for h in hs:
			lam0_mse = np.inf
			
			lam = abs(bests["lam"] - lam_step)
			while lam:
		
				print "hidden_layers:",h
				print "lambda:",lam


				regr = MLPRegressor(hidden_layer_sizes=h, activation=bests["activation"], alpha=lam, learning_rate="adaptive", max_iter=9001)
				regr.fit(X_train, y_train)
				y_pred = regr.predict(X_val)
				y_pred += y_mean

				y_pred = utils.unroll(y_pred, n_rolls=n_rolls)
				y_pred = y_pca.inverse_transform(y_pred)
				mse = utils.calc_rmse(y_pred, y_val)

				train_pred = regr.predict(X_train)

				train_pred += y_mean
				train_pred = utils.unroll(train_pred, n_rolls=n_rolls)
				train_pred = y_pca.inverse_transform(train_pred)

				train_mse = utils.calc_rmse(train_pred, y_train0)
				print "train_mse:", train_mse
				print "mse:",mse
				
				if mse + epsilon < bests["mse"] :
					bests["mse"] = mse
					bests["h"] = h
					bests["lam"] = lam
					cPickle.dump(bests, open("saves\\bests5.p", "wb"))
					updated = True

				lam = utils.gen_next_lambda(mse, train_mse, lam0_mse, lam, lam_step)
				lam0_mse = mse

		if updated:
			repeat = 0
		else:
			repeat += 1
		hs = utils.gen_new_h(bests["h"], repeat, magnitude=100, expand_out=expand_out)
		lam = bests["lam"] - lam_step
		
		print "new hs:", hs
		print "new lam:", lam
		print "current bests: "+str(bests)
	print str(bests)





def lasso(target_pca=True):
	from sklearn.decomposition import PCA
	from sklearn.linear_model import Lasso
		
	lams = [1e6,1e5,1e4,5e3,1e3,1e2,1e1,1e0,1e-1,1e-2,]
	X = np.loadtxt("train_X_ecog.csv", delimiter=",")
	y = np.loadtxt("train_Y_ecog.csv", delimiter=",")	

	tr_inds, val_inds = utils.split_data(X,y, group_size=200)
	X_train = X[tr_inds,:]
	y_train = y[tr_inds,:]
	X_val = X[val_inds,:]
	y_val = y[val_inds,:]

	if target_pca:
		y_pca = PCA(n_components=1)
		y_pca.fit(y_train)
		y_train = y_pca.transform(y_train)
		y_train.shape = (y_train.shape[0],)


	#X_train, X_val = utils.transform_data(X_train, X_val)
	X_train, X_val = utils.scale_data(X_train, X_val)

	best_mse = np.inf
	for lam in lams:
	
		print "lambda: ", lam
		regr = Lasso(alpha=lam)
		#y_pred = cross_val_predict(regr, X, y, cv=4, n_jobs=2)
		regr.fit(X_train, y_train)
		y_pred = regr.predict(X_val)
		if target_pca:
			y_pred.shape = ((y_pred.shape[0],1))
			y_pred = y_pca.inverse_transform(y_pred)
		mse = utils.calc_rmse(y_pred, y_val)
		print "mse:",mse
		if mse < best_mse:
			feature_importances = regr.coef_
			best_mse = mse
			best_lam = lam			

	print best_lam
	print best_mse
	from matplotlib import pyplot as plt
	feature_importances = abs(feature_importances / np.sum(feature_importances))
	plt.plot(np.arange(len(feature_importances)), feature_importances)
	plt.show()
	np.savetxt("saves\\lasso_coefs.csv", feature_importances)


def ridge(target_pca=True):
	from sklearn.decomposition import PCA
	from sklearn.linear_model import Ridge
		
	lams = [1e6,1e5,1e4,5e3,1e3,1e2,1e1,1e0]
	n_features = [5,8,10,12,15,20]
	X = np.loadtxt("train_X_ecog.csv", delimiter=",")
	y0 = np.loadtxt("train_Y_ecog.csv", delimiter=",")
	y0 -= y0.mean(axis=0)
	
	if target_pca:
		y_pca = PCA(n_components=1)
		y_pca.fit(y0)
		y = y_pca.transform(y0)
		y.shape = (y.shape[0],)
	else:
		y = y0.copy()

	tr_inds, val_inds = utils.split_data(X,y, group_size=200)
	X_train = X[tr_inds,:]
	y_train = y[tr_inds]
	X_val = X[val_inds,:]
	y_val = y0[val_inds,:]

	#X_train, X_val = utils.transform_data(X_train, X_val)
	X_train, X_val = scale(X_train), scale(X_val)

	best_mse = np.inf
	for n_f in n_features:
		for lam in lams:
			

			inds = utils.get_important_features(n_f)
			X_tr = X_train[:,inds]
			X_v = X_val[:,inds]	
			print "lambda: ", lam
			regr = Ridge(alpha=lam)
			#y_pred = cross_val_predict(regr, X, y, cv=4, n_jobs=2)
			regr.fit(X_tr, y_train)
			y_pred = regr.predict(X_v)
			if target_pca:
				y_pred.shape = ((y_pred.shape[0],1))
				y_pred = y_pca.inverse_transform(y_pred)
			mse = utils.calc_rmse(y_pred, y_val)
			print "mse:",mse
			if mse < best_mse:
				best_mse = mse
				best_lam = lam			
				best_n_features = n_f
	print best_n_features
	print best_lam
	print best_mse



def gb():
	from sklearn.decomposition import PCA
	from sklearn.ensemble import GradientBoostingRegressor
	X = np.loadtxt("train_X_ecog.csv", delimiter=",")
	y = np.loadtxt("train_Y_ecog.csv", delimiter=",")
	
	tr_inds, val_inds = utils.split_data(X,y, group_size=200)
	X_train = X[tr_inds,:]
	y_train = y[tr_inds,:]
	X_val = X[val_inds,:]
	y_val = y[val_inds,:]
	
	y_mean = y_train.mean(axis=0)
	y_train -= y_mean
	y_train = y_train.mean(axis=1)

	inds = utils.get_important_features(12)
	
	X_train0 = X_train[:,inds]
	X_val0 = X_val[:,inds]
	X_train = np.zeros((X_train0.shape[0], 1))
	X_val = np.zeros((X_val0.shape[0], 1))


	stacks = np.array([-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,])
	for i in stacks:

		X_train = np.hstack((X_train, np.roll(X_train0, -i, axis=0)))
		X_val = np.hstack((X_val, np.roll(X_val0, -i, axis=0)))

	X_train = np.delete(X_train, 0, axis=1)
	X_val = np.delete(X_val,0, axis=1)

	X_train = scale(X_train)
	X_val = scale(X_val)
	

	#learning_rates = [.03,.05,.07,.09,]
	#n_ests = [30,40,50,60]
	learning_rates = [.05]
	n_ests = [40]
	#max_ds = [4,5,6]
	max_ds = [6]
	best_mse = np.inf
	for r in learning_rates:
		for n in n_ests:
			for d in max_ds:
	
				print "learning rate:",r
				print "n_estimators:",n	
				print "max_depth:",d				
				regr = GradientBoostingRegressor(learning_rate=r, n_estimators=n, max_depth=d)
				#y_pred = cross_val_predict(regr, X, y, cv=4, n_jobs=2)
				regr.fit(X_train, y_train)
				y_pred = regr.predict(X_val)
				#y_pred.shape = ((y_pred.shape[0],1))
				#y_pred = y_pca.inverse_transform(y_pred)
				
				y_pred0 = np.zeros_like(y_val)
				for i in xrange(y_pred0.shape[1]):
					y_pred0[:,i] = y_pred
				y_pred0 += y_mean
				y_pred = y_pred0.copy()

				mse = utils.calc_rmse(y_pred, y_val)
				print "mse:",mse
				if mse < best_mse:
					feature_importances = regr.feature_importances_
					best_mse = mse
					best_r = r
					best_n = n
					best_d = d
					
	from matplotlib import pyplot as plt 
	print best_r
	print best_n
	print best_d
	print best_mse
	print inds
	imps = np.zeros(len(stacks))
	for i in xrange(len(stacks)):
		imps[i] = np.mean(feature_importances[12*i:12*i+12])
	plt.plot(stacks, imps)
	plt.ylabel("Importances")
	plt.xlabel("Time Step")
	plt.title("Time Step Importances")
	#plt.plot(np.arange(len(feature_importances)), feature_importances)
	plt.show()
	np.savetxt("saves\\feature_importances.csv", feature_importances)




def rf(source=None):
	from sklearn.ensemble import RandomForestRegressor
	#X, y = utils.load_data(source="rf_pca_X.csv", scaled=False, load_test=False)
	X = np.loadtxt("train_X_ecog.csv", delimiter=",")
	y = np.loadtxt("train_Y_ecog.csv", delimiter=",")
	#n_feats = ["sqrt", 40, 80, 120, 240, 417]
	#n_feats = [8,16,32,60,70]
	n_feats = [40,]
	y -= y.mean(axis=0)
	best_mse = np.inf
	
	tr_inds, val_inds = utils.split_data(X,y, group_size=200)
	X_train = X[tr_inds,:]
	y_train = y[tr_inds,:]
	X_val = X[val_inds,:]
	y_val = y[val_inds,:]

	#X_train, X_val = utils.transform_data(X_train, X_val)
	#X_train, X_val = scale(X_train), scale(X_val)

	for f in n_feats:
	
		print "max features:",f
				
		regr = RandomForestRegressor(n_estimators=100, max_features=f)
		#y_pred = cross_val_predict(regr, X, y, cv=4)
		regr.fit(X_train, y_train)
		y_pred = regr.predict(X_val)

		mse = utils.calc_rmse(y_pred, y_val)
		print "mse:",mse
		if mse < best_mse:
			best_mse = mse
			best_f = f
			feats = regr.feature_importances_

	print best_f
	print best_mse
	from matplotlib import pyplot as plt 
	plt.plot(np.arange(len(feats)), feats)
	plt.show()
	np.savetxt("saves\\rf_feature_importances.csv", feats)



def et(rollY=True):
	#d=21
	#l=7
	#mse=15.195

	from sklearn.ensemble import ExtraTreesRegressor
	X, y = utils.load_data(source="stacked_X.csv", load_test=False)
	depths = [11,15,21,29,35,40]
	min_leaves = [1,5,11,15,21]

	best_mse = np.inf
	for d in depths:
		for l in min_leaves:
			print "depth:",d
			print "min_leaf:",l
			regr = ExtraTreesRegressor(n_estimators=200, max_features=100, max_depth=d, min_samples_leaf=l, bootstrap=True, oob_score=True)

			if rollY: y = utils.roll_y(y)

			regr.fit(X, y)
			prediction = regr.oob_prediction_
			train_pred = regr.predict(X)
			if rollY:
				prediction = utils.unroll_y(prediction)
				train_pred = utils.unroll_y(train_pred)
				y = utils.unroll_y(y)

			mse = utils.calc_rmse(y, prediction)
			train_mse = utils.calc_rmse(y, train_pred)
			print "train_mse:", train_mse
			print "mse:",mse
			if mse < best_mse:
				best_mse = mse
				best_d = d
				best_l = l
	print best_d
	print best_l
	print best_mse




