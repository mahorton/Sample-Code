import numpy as np
from scipy import stats as st
from scipy import optimize as op
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
import pandas as pd


titanic_data = pd.read_excel('titanic3.xls', sheetname=0, parse_cols=6)
titanic_data = titanic_data.drop('name', axis=1)

titanic_data = titanic_data[np.isfinite(titanic_data['age'])]

titanic_data['sex'][titanic_data['sex'] == 'male'] = 1
titanic_data['sex'][titanic_data['sex'] == 'female'] = 0




natural_data = titanic_data.copy()

titanic_data["pclass2"] = 0
pclass1 = titanic_data["pclass"] == 1
pclass2 = titanic_data["pclass"] == 2
pclass3 = titanic_data["pclass"] == 3
titanic_data["pclass"][pclass1] = 1
titanic_data["pclass2"][pclass1] = 0
titanic_data["pclass"][pclass2] = 0
titanic_data["pclass2"][pclass2] = 1
titanic_data["pclass"][pclass3] = 0
titanic_data["pclass2"][pclass3] = 0

#switches the order of the colums
cols = titanic_data.columns.tolist()
order = [0, 6, 2, 3, 4, 5, 1]
cols = [cols[i] for i in order]
titanic_data = titanic_data[cols]

cols = natural_data.columns.tolist()
order = [0, 2, 3, 4, 5, 1]
cols = [cols[i] for i in order]
natural_data = natural_data[cols]
print cols
#create the training and test sets
N_train = int(.6 * titanic_data.shape[0])
perm = np.random.permutation(titanic_data.values)
training = perm[:N_train].astype(float)
test = perm[N_train:].astype(float)
natural_perm = np.random.permutation(natural_data.values)
natural_training = natural_perm[:N_train].astype(float)
natural_test = natural_perm[N_train:].astype(float)
#training_ind = np.random.choice(titanic_data.shape[0], N_train, False)
#training = titanic_data.values[training_ind, :]


m = LogisticRegression(C = 1)
m.fit(natural_training[:,:-1], natural_training[:,-1])
probs = m.predict_proba(natural_test[:,:-1])
fpr, tpr, thresh = roc_curve(natural_test[:,-1], probs[:,1], pos_label=1)

nb = MultinomialNB()
nb.fit(natural_training[:,:-1], natural_training[:,-1])
nb_probs = nb.predict_proba(natural_test[:,:-1])
nb_fpr, nb_tpr, nb_thresh = roc_curve(natural_test[:,-1], nb_probs[:,1], pos_label=1)

mm = LogisticRegression(C = 1)
mm.fit(training[:,:-1], training[:,-1])
probs2 = mm.predict_proba(test[:,:-1])
fpr2, tpr2, thresh2 = roc_curve(test[:,-1], probs2[:,1], pos_label=1)

plt.plot(fpr, tpr, label="Logistic Classifier for pclass = {1,2,3}")
plt.plot(fpr2, tpr2, label="Logistic Classifier")
plt.plot(nb_fpr, nb_tpr, label="Naive Bayes Classifier")
plt.legend(loc=4)
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()

#print titanic_data
print "Logistic Classifier for pclass = {1,2,3}", auc(fpr, tpr)
print "Logistic AUC score:", auc(fpr2,tpr2)
print "Naive Bayes AUC score:", auc(nb_fpr, nb_tpr)

print "coefs", m.coef_
print "coefs", nb.coef_
