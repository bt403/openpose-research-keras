#ONLY HOGS 8 - Smaller - 2880 epochs
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from numpy.random import seed
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score, cross_validate
from tensorflow.keras.models import Model, Sequential
from keras.layers import Dense, Activation, Input, Layer, Dropout, Concatenate
from keras import regularizers
from util_autoencoder_training import *
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--train_data', type=str, default='./data/estimates/train_data.pkl')
parser.add_argument('--train_data_hog', type=str, default='./data/estimates/train_data_hog.pkl')
parser.add_argument('--test_data', type=str, default='./data/estimates/test_data.pkl')
parser.add_argument('--test_data_hog', type=str, default='./data/estimates/test_data_hog.pkl')
args = parser.parse_args()

train = pd.read_pickle(args.train_data)
test = pd.read_pickle(args.test_data)
hog_features_d = pd.read_pickle(args.train_data_hog)
hog_features_d_test = pd.read_pickle(args.test_data_hog)
columns_filter= [x for x in train.columns if x not in ["video_name", "coordX", "coordY", "size", "orientation", "path", "path_cropped", "target", "ref_dist", "ears", "neck", "mouth", "cheeks", "eyes", "nose", "forehead"]]

seed(1)
tf.random.set_seed(1)

targets = [["mouth", "cheeks", "eyes", "nose", "ears"]]
targets = ["target"]

geometrical_dim = len(columns_filter)
latent_dim = 24
latent_dim_hog = 1024
input_dim_hog = len(hog_features_d.iloc[0])
input_dim = geometrical_dim

#Model Base
encoder_inputs_base = Input(shape=(input_dim,))
x_base = Dropout(0.5, input_shape=(input_dim,))(encoder_inputs_base)
x_base = Dense(128, activation='relu', input_shape=(input_dim,))(x_base)
x_base = Dense(64, activation='relu')(x_base)
x_base = Dense(latent_dim, name='encoder_output', activity_regularizer=regularizers.l1(1.0))(x_base)
z_mean_base = Dense(latent_dim, name="z_mean")(x_base)
z_log_var_base = Dense(latent_dim, name="z_log_var")(x_base)
z_base = x_base

latent_inputs_base = Input(shape=(latent_dim,))
x_base = Dense(64, activation='relu', input_shape=(latent_dim,))(latent_inputs_base)
x_base = Dense(128, activation='relu')(x_base)
decoder_outputs_base = Dense(input_dim, activation=None)(x_base)


#Model with HOG
latent_dim_hog = 8
encoder_inputs_hog = Input(shape=(input_dim_hog,))
encoder_inputs_geometrical = Input(shape=(input_dim,))
x_hog = Dropout(0.5, input_shape=(input_dim,))(encoder_inputs_hog) 
x_hog = Dense(1024, activation='relu')(x_hog)
x_hog = Dense(512, activation='relu')(x_hog)
x_hog = Dense(latent_dim_hog, name='encoder_output_hog', activity_regularizer=regularizers.l1(1.0))(x_hog)

encoder_inputs_geometrical = Input(shape=(input_dim,))
x_geometrical = Dropout(0.5)(encoder_inputs_geometrical) 
x_geometrical = Dense(128, activation='relu', input_shape=(input_dim,))(x_geometrical)
x_geometrical = Dense(64, activation='relu')(x_geometrical)
x_geometrical = Dense(latent_dim, name='encoder_output_geometrical', activity_regularizer=regularizers.l1(1.0))(x_geometrical)

x_geometrical_hog = Concatenate()([x_geometrical, x_hog])

z_mean_hog = Dense(latent_dim, name="z_mean")(x_geometrical_hog)
z_log_var_hog = Dense(latent_dim, name="z_log_var")(x_geometrical_hog)
z_hog = x_geometrical_hog
latent_inputs_hog_geometrical = Input(shape=(latent_dim + latent_dim_hog,))
x_geometrical_hog = Dense(512+64, activation='relu', input_shape=(latent_dim,))(latent_inputs_hog_geometrical)
x_geometrical_hog = Dense(1024+128, activation='relu')(x_geometrical_hog)
decoder_outputs_hog = Dense(input_dim_hog+input_dim, activation=None)(x_geometrical_hog)

opt_hog = keras.optimizers.Adam(learning_rate=0.0005)
opt = keras.optimizers.Adam(learning_rate=0.000001, clipnorm=1.0, clipvalue=1.0)


for i in targets:
  print("------")
  print("RESULTS FOR: " + str(i))
  inverse = i != "target"
  binary = True
  weights = "balanced"
  #############

  results = {"model_001": [], "model_002": []}
  if not isinstance(i, str):
    for k in results.keys():
      results[k] = {}
      for t in i:
        results[k][t] = []

  if not isinstance(i, str):
    train = train[train["target"] == "on-head"]
    test = test[test["target"] == "on-head"]
    binary = False

  encoder_hog = Model([encoder_inputs_geometrical, encoder_inputs_hog], [z_mean_hog, z_log_var_hog, z_hog], name="encoder")
  decoder_hog = Model(latent_inputs_hog_geometrical, decoder_outputs_hog, name="decoder")

  encoder_base = Model(encoder_inputs_base, [z_mean_base, z_log_var_base, z_base], name="encoder")
  decoder_base = Model(latent_inputs_base, decoder_outputs_base, name="decoder")

  autoencoder_hog = VAE(encoder_hog, decoder_hog)
  autoencoder_hog.compile(optimizer=opt_hog)

  autoencoder_base = VAE(encoder_base, decoder_base)
  autoencoder_base.compile(optimizer=opt)

  runCrossValidationAutoencoderSVM_withHOG(results, train, hog_features_d, test, hog_features_d_test, i, "model_002", autoencoder_hog, encoder_hog, decoder_hog, use_hog=True, weights=weights, binary=binary)
  runCrossValidationAutoencoderSVM(results, train, hog_features_d, test, hog_features_d_test, i, "model_001", autoencoder_base, encoder_base, decoder_base, weights=weights, binary=binary)

  for k in results.keys():
    df_results = pd.DataFrame(results[k])
    print("------")
    print("Results for " + k + "")
    display(df_results)
