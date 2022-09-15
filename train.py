# Numpy, pandas, time
import numpy as np
import pandas as pd
import time

# Iterative Stratification untuk cross validation multilabel
from skmultilearn.model_selection import IterativeStratification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

#import models
from src.sae_dnn_model import sae_model, dnn_model
from src.iterative_strat_modif import new_init
from src.visualize import visual

#Modifikasi IterativeStratification agar hasil random data tetap sama    
IterativeStratification.__init__ = new_init
# define evaluation procedure
np.random.seed(123)
#Inisialisasi CV
cv = IterativeStratification(n_splits=5, random_state = 123)

#Main Code 
def res_sae_dnn(X, hl_node, lr, opt, num_layers, do, fr_node):
  
  acc_results = list()
  f1_results = list()
  prec_results = list()
  rec_results = list()
  n_inputs, n_outputs = X.shape[1], 7
  
  print("finding sae weights....")
  ti0 = time.time()

  sae_weigths_tuned = sae_model(xt = X, hl_node = hl_node, af = "relu", lr = lr, opt= opt, num_layers = num_layers,
                                do = do, fr_node= fr_node)

  ti1 = time.time()
  print('done, processing time:', ti1-ti0)
  
  i=0
  t0 = time.time()
  # enumerate folds
  np.random.seed(123)

  for train_ix, test_ix in cv.split(X,Y):
    X_train, X_test = X.iloc[train_ix,:], X.iloc[test_ix,:]
    y_train, y_test = Y[train_ix], Y[test_ix]
    # define model
    model = dnn_model(xt = X_train, sae_weights = sae_weigths_tuned, hl_node = hl_node, af = "relu", lr = lr, opt= opt, num_layers = num_layers,
                              do = do, fr_node= fr_node)
    # fit model
    model.fit(X_train, y_train, verbose=False, epochs=100)
    # make a prediction on the test set
    yhat = model.predict(X_test)
    # round probabilities to class labels
    yhat = yhat.round()
    # calculate metrics
    acc = accuracy_score(y_test, yhat)
    f1 = f1_score(y_test, yhat, average='samples')
    prec = precision_score(y_test, yhat, average='samples' ,zero_division=0)
    rec = recall_score(y_test, yhat, average='samples')
    
    # store result
    print("CV number: ", i)
    print('accuracy of :>%.3f' % acc)
    print('F1 of :>%.3f' % f1)
    print('Precision of :>%.3f' % prec)
    print('Recall of :>%.3f' % rec)
    acc_results.append(acc)
    f1_results.append(f1)
    prec_results.append(prec)
    rec_results.append(rec)
    i=i+1
  
  t1 = time.time()
  total_waktu = t1-t0

  print("waktu proses: ", total_waktu)
  print("Accuracy array:", acc_results)
  print("F1 array:", f1_results)
  print("Precision array:", prec_results)
  print("Recall array:", rec_results)
  
  print('SAE-DNN TUNED PERFORMANCE')
  print('Accuracy    : {0:.5f}±{1:.3f}'.format(np.mean(acc_results), np.std(acc_results)))
  print('F1 Score    : {0:.5f}±{1:.3f}'.format(np.mean(f1_results), np.std(f1_results)))
  print('Precision   : {0:.5f}±{1:.3f}'.format(np.mean(prec_results), np.std(prec_results)))
  print('Recall      : {0:.5f}±{1:.3f}'.format(np.mean(rec_results), np.std(rec_results)))
  
  accuracy_res, f1_res, precision_res, recall_res = np.mean(acc_results), np.mean(f1_results), np.mean(prec_results), np.mean(rec_results)
  metric_sae_dnn = [accuracy_res, f1_res, precision_res, recall_res]

  model.save("models/sae_dnn_pubchem_tuned.h5")

  return [metric_sae_dnn, total_waktu]


def dnn_saja(X, hl_node, lr, opt, num_layers, do, fr_node):
  acc_results = list()
  f1_results = list()
  prec_results = list()
  rec_results = list()
  n_inputs, n_outputs = X.shape[1], 7
  # define evaluation procedure
  # cv = IterativeStratification(n_splits=5, random_state = 123)
  i=0
  # enumerate folds
  t0 = time.time()

  for train_ix, test_ix in cv.split(X,Y):
    X_train, X_test = X.iloc[train_ix,:], X.iloc[test_ix,:]
    y_train, y_test = Y[train_ix], Y[test_ix]
    # define model tanpa bobot SAE
    model = dnn_model(xt= X_train, sae_weights= None, hl_node= hl_node, af= "relu", lr= lr, opt= opt, num_layers= num_layers,
                              do= do, fr_node= fr_node)
    # fit model
    model.fit(X_train, y_train, verbose=False, epochs=100)
    # make a prediction on the test set
    yhat = model.predict(X_test)
    # round probabilities to class labels
    yhat = yhat.round()
    # calculate metrics
    acc = accuracy_score(y_test, yhat)
    f1 = f1_score(y_test, yhat, average='samples')
    prec = precision_score(y_test, yhat, average='samples', zero_division=0)
    rec = recall_score(y_test, yhat, average='samples')
    # store result
    print("CV number: ", i)
    print('accuracy of :>%.3f' % acc)
    print('F1 of :>%.3f' % f1)
    print('Precision of :>%.3f' % prec)
    print('Recall of :>%.3f' % rec)
    acc_results.append(acc)
    f1_results.append(f1)
    prec_results.append(prec)
    rec_results.append(rec)
    i=i+1
  
  t1 = time.time()
  total_waktu = t1-t0
  print("waktu proses", total_waktu)
  print("Accuracy array:", acc_results)
  print("F1 array:", f1_results)
  print("Precision array:", prec_results)
  print("Recall array:", rec_results)
  
  print('DNN TUNED PERFORMANCE')
  print('Accuracy    : {0:.5f}±{1:.3f}'.format(np.mean(acc_results), np.std(acc_results)))
  print('F1 Score    : {0:.5f}±{1:.3f}'.format(np.mean(f1_results), np.std(f1_results)))
  print('Precision   : {0:.5f}±{1:.3f}'.format(np.mean(prec_results), np.std(prec_results)))
  print('Recall      : {0:.5f}±{1:.3f}'.format(np.mean(rec_results), np.std(rec_results)))
  
  accuracy_res, f1_res, precision_res, recall_res = np.mean(acc_results), np.mean(f1_results), np.mean(prec_results), np.mean(rec_results)
  metric_sae_dnn = [accuracy_res, f1_res, precision_res, recall_res]

  model.save("models/dnn_only_pubchem_tuned.h5")

  return [metric_sae_dnn, total_waktu]

#datasets
df_pubchem = pd.read_csv('dataset/df_pubchem_rapi.csv')
Y = pd.read_csv('dataset/kelas_data.csv')
Y = np.array(Y)

#Parameter
X = df_pubchem
hl_node = 1800
fr_node = 0.66
num_layers = 2
opt = "adam"
lr = 0.1
do = 0.5

#train
hasil_saednn = res_sae_dnn(X, hl_node, lr, opt, num_layers, do, fr_node)
dnn_aja = dnn_saja(X, hl_node, lr, opt, num_layers, do, fr_node)

print("SAE-DNN training time:" + str(hasil_saednn[1]) + " seconds")
print("DNN Only training time: " + str(dnn_aja[1]) + " seconds")

#visualize saednn vs dnnonly
path = 'reports/figure'
visual(hasil_saednn[0], dnn_aja[0], "SAE-DNN vs DNN Only", path)



