from math import sqrt
import numpy as np
from numpy import concatenate
from numpy import array
from numpy import mean
from numpy import std
from scipy import interp
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import statistics
from statistics import mean 
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib import pyplot
fig, ax = plt.subplots()
from itertools import cycle
import tensorflow
import keras
from keras import layers
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense
from keras import backend as K
from keras.models import Model
from keras.models import Sequential
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
import sklearn
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import auc
import imblearn
from imblearn.over_sampling import SMOTE
import shap

##set seed for reproducibility
seed = 0

##set numpy seed
numpy.random.seed(seed)

##function to calculate mean
def calculate_mean(lst): 
    return mean(lst)

##function to calculated sd
def calculate_sd(lst):
  return statistics.pstdev(lst) 

##function that converts a csv file to a dataframe
def csvDf(dat, **kwargs): 
  data = array(dat)
  if data is None or len(data)==0 or len(data[0])==0:
    return None
  else:
    return pd.DataFrame(data[1:,1:],index=data[1:,0],columns=data[0,1:],**kwargs)

##function for AUC score
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

##function for specificity score
def specificity(y_true, y_pred):
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fp = K.sum(neg_y_true * y_pred)
    tn = K.sum(neg_y_true * neg_y_pred)
    specificity = tn / (tn + fp + K.epsilon())
    return specificity

##function for precision score
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

##function to prepare data
def prepare_data(): 
  dataset = pd.read_csv('')
  values = dataset.values
  dataset = csvDf(dataset)
  shifted = dataset[dataset.columns[-1:]].groupby(level=0).shift(-3)
  dataset[-1] = shifted
  df = dataset.dropna()
  X = df[df.columns[:-1]].to_numpy().astype(float)
  Y = df[df.columns[-1:]].to_numpy().astype(float)
  return np.asarray(X).astype(float), np.asarray(Y).astype(float)

##function that returns a keras model
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

##function to balance data
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

##arrays to keep track of scores
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

##function to run keras model
def run_keras_model(X, y):
  for train, test in kfold.split(X, y):
      model = get_model()
      oversample = SMOTE()
      xtrain, ytrain = oversample.fit_resample(np.asarray(X[train]), np.asarray(y[train]))
      xtrain = preprocessing.scale(xtrain)
      xtest = preprocessing.scale(X[test])
      model.fit(np.asarray(xtrain), np.asarray(ytrain), callbacks = callbacks, epochs=1000, batch_size=64, verbose=1, shuffle = True)
      tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y[test], model.predict(X[test]).round()).ravel()
      scores = model.evaluate(X[test], y[test])
      score1.append((tp +tn)/(tp+tn+fp+fn))
      score2.append(tp/(tp+fn))
      score3.append(tn/(tn+fp))
      score4.append(tp/(tp+fp))
      score5.append(2*((prec*sens)/(prec+sens)))
      tps.append(tp)
      fps.append(fp)
      fns.append(fn)
      tns.append(tn)

  print_results(score1, score2, score3, score4, score5, tps, tns, fps, fns)

##function that prints the results to the console
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
  
##function that computes predictor importance
def compute_predictor_importance():
  shap.initjs()
  explainer = shap.KernelExplainer(model.predict_proba, test_X[0:100,:])
  shap_values = explainer.shap_values(test_X[0:100,:])
  shap.summary_plot(shap_values, test_X[0:100,:], plot_type="bar")

##function to run sklearn models
def run_sklearn_model_comp(X,y):
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
        oversample = SMOTE()
        xtrain, ytrain = oversample.fit_resample(np.asarray(X[train]), np.asarray(y[train]))
        xtrain = preprocessing.scale(xtrain)
        xtest = preprocessing.scale(X[test])
        model.fit(xtrain, np.asarray(ytrain).ravel())
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y[test].ravel(), model.predict(xtest).round()).ravel()
        score1.append((tp +tn)/(tp+tn+fp+fn))
        score2.append(tp/(tp+fn))
        score3.append(tn/(tn+fp))
        score4.append(tp/(tp+fp))
        score5.append(2*((prec*sens)/(prec+sens)))
        tps.append(tp)
        fps.append(fp)
        fns.append(fn)
        tns.append(tn)
        
      print_results(score1, score2, score3, score4, score5, tps, tns, fps, fns)

##function to plot ROC curves
def plot_ROC_curves():
    cv = StratifiedKFold(n_splits=10)
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

##main function
def main():
  X, y = prepare_data()
  run_sklearn_model_comp(X,y) ##runs model comparision
  run_keras_model(np.asarray(X),np.asarray(y)) ##runs keras model
  compute_predictor_importance() ##computer predictor importance
    


if __name__ == "__main__:":
  main()

##declare kfold var
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

##declare keras early stopping var
callbacks = [EarlyStopping(monitor='loss', patience=2),
              ModelCheckpoint(filepath='best_model.h5', monitor='loss', save_best_only=True)]

##call main function to run program
main()
