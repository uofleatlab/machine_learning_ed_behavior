from math import sqrt
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
import numpy
import numpy as np
from numpy import concatenate
from scipy import interp
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import pandas as pd
from statistics import mean 
import statistics
import matplotlib.pyplot as plt
from matplotlib import pyplot
fig, ax = plt.subplots()
from itertools import cycle
import sklearn
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import auc
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import tensorflow
import tensorflow as tf
import keras
from keras import layers
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense
from keras import backend as K
from keras.models import Model
from keras.models import Sequential
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import shap

seed = 0
# set other seeds
numpy.random.seed(seed)

def calculate_mean(lst): 
    return mean(lst)

def calculate_sd(lst):
  return statistics.pstdev(lst) 

def csvDf(dat,**kwargs): 
  from numpy import array
  data = array(dat)
  if data is None or len(data)==0 or len(data[0])==0:
    return None
  else:
    return pd.DataFrame(data[1:,1:],index=data[1:,0],columns=data[0,1:],**kwargs)

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def specificity(y_true, y_pred):
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fp = K.sum(neg_y_true * y_pred)
    tn = K.sum(neg_y_true * neg_y_pred)
    specificity = tn / (tn + fp + K.epsilon())
    return specificity

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
    
def prepare_data():

  #dataset = pd.read_csv('/content/gdrive/My Drive/binge_filled_final.csv')
  #dataset = pd.read_csv('/content/gdrive/My Drive/restriccc 2 2.csv')
  #dataset = pd.read_csv('/content/gdrive/My Drive/purge_ts.csv')  
  dataset = pd.read_csv('/content/gdrive/My Drive/purge_nts.csv')
  #dataset = pd.read_csv('/content/gdrive/My Drive/restriccc 2 2_one.csv')

  print(dataset)
  values = dataset.values

  dataset = csvDf(dataset)

  #dataset = dataset[dataset.columns[:-1]]

  #shifted = dataset[dataset.columns[-1:]].groupby(level=0).shift(-3)

  #print(dataset)
  #print(shifted)


  #dataset[-1] = shifted

  #df = dataset.dropna()
  #print(df)
  df = dataset

 # dat = dataset.to_numpy()
  

 # Y = dat[:, -1] # for last column
 # X = dat[:, :-1] # for all but last column


  X = df[df.columns[:-1]].to_numpy().astype(float)
  Y = df[df.columns[-1:]].to_numpy().astype(float)
  print(X)
  print(Y)

  
  return np.asarray(X).astype(float), np.asarray(Y).astype(float)
  
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

callbacks = [EarlyStopping(monitor='loss', patience=2),
              ModelCheckpoint(filepath='best_model.h5', monitor='loss', save_best_only=True)]
              
def get_model():
  model = Sequential()
  model.add(Dense(56, activation='relu'))
  model.add(layers.BatchNormalization())
  #model.add(keras.layers.Dropout(0.2))
  model.add(Dense(28, activation='relu'))
  model.add(layers.BatchNormalization())
  #model.add(keras.layers.Dropout(0.2))
  model.add(Dense(1, activation='sigmoid'))
  #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)   
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', precision, specificity])
  return model



def balanced_subsample(x,y,subsample_size=1.0):

    xs = []
    ys = []
    countmaj = 0
    countmin = 0
    iter = 0

    for i in y:
      if i == 1:
        countmin = countmin + 1
        xs.append(x[iter])
        ys.append(i)
      else:
        if countmin > countmaj:
          countmaj = countmaj + 1
          xs.append(x[iter])
          ys.append(i)
         
      iter = iter + 1    

    return xs,ys

tprs = []
aucs = []
score1 = []
score2 = []
score3 = []
score4 = []
score5 = []
tps = []
tns = []
fps = []
fns = []

def cross_validate(X, y):
  for train, test in kfold.split(X, y):
      model = get_model()
      X[train] = preprocessing.normalize(X[train])
      X[test] = preprocessing.normalize(X[test])

      Xtr, ytr = balanced_subsample(np.asarray(X[train]), np.asarray(y[train]))


     
      model.fit(np.asarray(Xtr), np.asarray(ytr), callbacks = callbacks, epochs=1000, batch_size=64, verbose=1, shuffle = True)
      tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y[test], model.predict(X[test]).round()).ravel()
      accuracy = (tp +tn)/(tp+tn+fp+fn)
      sens = tp/(tp+fn)
      spec = tn/(tn+fp)
      prec = tp/(tp+fp)
      f1 = 2*((prec*sens)/(prec+sens))
      scores = model.evaluate(X[test], y[test])
      score1.append(accuracy)
      score2.append(sens)
      score3.append(spec)
      score4.append(prec)
      score5.append(f1)
      tps.append(tp)
      fps.append(fp)
      fns.append(fn)
      tns.append(tn)

  print_results(score1, score2, score3, score4, score5, tps, tns, fps, fns)

def print_results(score1, score2, score3, score4, score5, tps, tns, fps, fns):
  print("Accuracy: %0.2f (+/- %0.2f)" % (calculate_mean(score1), calculate_sd(score1) * 2))
  print("Sensitivity: %0.2f (+/- %0.2f)" % (calculate_mean(score2), calculate_sd(score2) * 2))
  print("Specificity: %0.2f (+/- %0.2f)" % (calculate_mean(score3), calculate_sd(score3) * 2))
  print("Precision: %0.2f (+/- %0.2f)" % (calculate_mean(score4), calculate_sd(score4) * 2))
  print("F1: %0.2f (+/- %0.2f)" % (calculate_mean(score5), calculate_sd(score5) * 2))

  print("TP: %0.2f (+/- %0.2f)" % (calculate_mean(tps), calculate_sd(tps) * 2))
  print("TN: %0.2f (+/- %0.2f)" % (calculate_mean(tns), calculate_sd(tns) * 2))
  print("FP: %0.2f (+/- %0.2f)" % (calculate_mean(fps), calculate_sd(fps) * 2))
  print("FN: %0.2f (+/- %0.2f)" % (calculate_mean(fns), calculate_sd(fns) * 2))
  
def compute_feature_importance():
  shap.initjs()
  explainer = shap.KernelExplainer(model.predict_proba, test_X[0:100,:])
  shap_values = explainer.shap_values(test_X[0:100,:])
  shap.summary_plot(shap_values, test_X[0:100,:], plot_type="bar")


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import imblearn
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
import numpy as np


def ru(X,y):
    h = .02  # step size in the mesh

    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
            "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
            "Naive Bayes", "QDA"]

    classifiers = [
        DecisionTreeClassifier(max_depth=10, class_weight='balanced'),
        RandomForestClassifier(max_depth=10, n_estimators=10, max_features=1, class_weight='balanced'),
        MLPClassifier(alpha=1, max_iter=10000),
        SVC(kernel="linear", C=0.025),
        sklearn.linear_model.LogisticRegression(max_iter=10000)]


    for name, clf in zip(names, classifiers):

      tprs = []
      aucs = []
      score1 = []
      score2 = []
      score3 = []
      score4 = []
      score5 = []
      tps = []
      tns = []
      fps = []
      fns = []

      for train, test in kfold.split(X, y):
        model = clf



        # transform the dataset
        oversample = SMOTE()
        Xtr, ytr = oversample.fit_resample(np.asarray(X[train]), np.asarray(y[train]))
        Xtr = preprocessing.scale(Xtr)
        Xte = preprocessing.scale(X[test])


        model.fit(Xtr, np.asarray(ytr).ravel())
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y[test].ravel(), model.predict(Xte).round()).ravel()
        accuracy = (tp +tn)/(tp+tn+fp+fn)
        sens = tp/(tp+fn)
        spec = tn/(tn+fp)
        prec = tp/(tp+fp)
        f1 = 2*((prec*sens)/(prec+sens))
        score1.append(accuracy)
        score2.append(sens)
        score3.append(spec)
        score4.append(prec)
        score5.append(f1)
        tps.append(tp)
        fps.append(fp)
        fns.append(fn)
        tns.append(tn)

      print_results(score1, score2, score3, score4, score5, tps, tns, fps, fns)
      
   import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import StratifiedKFold


# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=5)

classifier = MLPClassifier(alpha=1, learning_rate_init = 0.01, max_iter=10000)
model = classifier

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots()

X, y = prepare_data()

for i, (train, test) in enumerate(kfold.split(X, y)):
    classifier.fit(X[train], y[train])
    viz = plot_roc_curve(classifier, X[test], y[test],
                         alpha=0.3, lw=1)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="Receiver operating characteristic example")
ax.legend(loc="lower right")
plt.show()

def main():
  X, y = prepare_data()
  '''
  # define the model
  model = GaussianNB()


  # evaluate the model
  cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
  n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
  # report performance
  
  print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
  y_pred = cross_val_predict(model, X, y, cv=10)
  
  
  tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
  print(tn)
  print(tp)
  print(fp)
  print(fn)
  sens = tp/(tp+fn)
  spec = tn/(tn+fp)
  prec = tp/(tp+fp)
  f1 = 2*((prec*sens)/(prec+sens))

  print(sens)
  print(spec)
  print(prec)
  print(f1)
  '''
 # ru(X,y)
  cross_validate(np.asarray(X),np.asarray(y))
  compute_feature_importance()


if __name__ == "__main__:":
  main()
  
main()
