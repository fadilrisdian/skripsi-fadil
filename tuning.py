from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import tensorflow as tf
import kerastuner as kt
from keras_tuner import BayesianOptimization
from tensorflow.keras.callbacks import  EarlyStopping

from src.sae_dnn_model import sae_model, dnn_model


#Fungsi model untuk tuning
def build_model(hp):
    #Isi parameter yang akan dituning
    params = {
              'hl_node' : hp.Choice('units',values= para_hl_node),
              'af' : hp.Choice('activation',values= para_af),
              'lr' : hp.Choice('learning_rate',values= para_lr),
              'opt' : hp.Choice('optimizer',values= para_opt),
              'num_layers' : hp.Choice('num_layers',values= para_num_layers),
              'do' : hp.Choice('dropout_rate',values= para_do),
              'fr_node' : hp.Choice('fraction_node',values= para_fr_node)
              }
    #Latih model SAE
    sae_weights = sae_model(xt = X, xv = X_train, EPOCHS= 100,**params)
    #Latih model DNN dengan bobot SAE
    sae_dnn = dnn_model(X_train, sae_weights=sae_weights, EPOCHS= 100,**params)
    return sae_dnn


#Read
X = pd.read_csv('dataset/df_pubchem_rapi.csv')
Y = pd.read_csv('dataset/kelas_data.csv')
Y = np.array(Y)

para_hl_node = [320, 640, 1280, 1500, 1600, 1700, 1800]
para_af = ["relu"]
para_lr = [x for x in np.linspace(0.01,0.1)]
para_opt = ["adam", "adagrad"]
para_num_layers = [2,3,4,5]
para_do = [0.5,0.6,0.7,0.8]
para_fr_node = [0.5,0.66,0.75]

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2)
#hypterparameter tuning
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_recall',patience = 25)

# Fungsi Bayesian Optimization di Keras Tuner.
tuner = BayesianOptimization(build_model,
    # Metrik yang dicari optimalnya
    objective= kt.Objective("val_recall", direction="max"), 
    # Jumlah percobaan
    max_trials=2,
    executions_per_trial=2,
    # Folder simpan hasil tuning
    directory='tuning-model',
    project_name='sae_dnn_tuning_bayes', overwrite = True)

#Jalankan Keras Tuner u
tuner.search(X_train, y_train, epochs=100, validation_data=(X_test, y_test),callbacks=[stop_early])
#Tampilkan hasil terbaik
tuner.results_summary()