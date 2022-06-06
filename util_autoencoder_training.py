from keras import metrics, losses
from tensorflow.keras.models import Model, Sequential
from keras.layers import Dense, Activation, Input, Layer, Dropout, Concatenate
from keras import regularizers
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from util_classifier_training import *

class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            print ("encoding")
            z_mean, z_log_var, z = self.encoder(data)
            print("decoding")
            reconstruction = self.decoder(z)
            print("decoded")
            mse = tf.keras.losses.MeanSquaredError()
            if tf.is_tensor(data[0]):
              reconstruction_loss =mse(data, reconstruction)
            else:
              reconstruction_loss =mse(Concatenate()([data[0][0], data[0][1]]), reconstruction)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))*0
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def runCrossValidationAutoencoderSVM_withHOG(results, x_data, x_data_hog, x_data_test, x_data_hog_test, target_val, model_name, model_autoencoder, model_encoder, model_decoder, use_hog = False, weights="balanced", binary=True):
  print("training model: " + model_name)
  columns_filter= [x for x in x_data.columns if x not in ["video_name", "coordX", "coordY", "size", "orientation", "path", "path_cropped", "target", "ref_dist", "ears", "neck", "mouth", "cheeks", "eyes", "nose", "forehead"]]

  for i in columns_filter:
    x_data[i].fillna(x_data[i].median(), inplace=True)
    x_data_test[i].fillna(x_data[i].median(), inplace=True)

  columns_filter= [x for x in x_data.columns if x not in ["video_name", "coordX", "coordY", "size", "orientation", "path", "path_cropped", "target", "ref_dist", "ears", "neck", "mouth", "cheeks", "eyes", "nose", "forehead"]]

  x_train_hog = pd.concat([x_data_hog], axis=1).astype('float32')
  x_test_hog = pd.concat([x_data_hog_test], axis=1).astype('float32')
  
  x_train_2 = pd.DataFrame(x_data[columns_filter]).astype('float32')
  x_test_2 = pd.DataFrame(x_data_test[columns_filter]).astype('float32')

  scaler = StandardScaler()
  scaler_hog = StandardScaler()

  x_train_2[columns_filter] = scaler.fit_transform(x_train_2[columns_filter])
  x_test_2[columns_filter] = scaler.transform(x_test_2[columns_filter])

  #x_train_hog = scaler_hog.fit_transform(x_train_hog)
  #x_test_hog = scaler_hog.transform(x_test_hog)

  total_epochs_per_repeat = 20

  for i in range(5):
    print("---- Repeat number " + str(i+1) + " ----")
    model_history = model_autoencoder.fit([x_train_2, x_train_hog], epochs=total_epochs_per_repeat, batch_size=32, verbose=1, workers=4,)

    print("---- Calculating SVM - Repeat number " + str(i+1) + " ----")
    x_train = model_encoder([np.array(x_train_2), np.array(x_train_hog)])[2]
    x_test = model_encoder([np.array(x_test_2), np.array(x_test_hog)])[2]

    params = getBestParams(results, x_train.numpy(), x_data[target_val], with_pca=False, binary=binary, groups=x_data["video_name"])
    train_svm(results, x_train.numpy(), x_data[target_val], x_test, x_data_test[target_val], model_name, params, x_data["video_name"], weights, False, binary)


def runCrossValidationAutoencoderSVM(results, x_data, x_data_hog, x_data_test, x_data_hog_test, target_val, model_name, model_autoencoder, model_encoder, model_decoder, use_hog = False, weights="balanced", binary=True):
  print("training model: " + model_name)
  columns_filter= [x for x in x_data.columns if x not in ["video_name", "coordX", "coordY", "size", "orientation", "path", "path_cropped", "target", "ref_dist", "ears", "neck", "mouth", "cheeks", "eyes", "nose", "forehead"]]
  print(columns_filter)
  for i in columns_filter:
    x_data[i].fillna(x_data[i].median(), inplace=True)
    x_data_test[i].fillna(x_data[i].median(), inplace=True)

  if (use_hog):
    x_train_2 = pd.concat([x_data_hog, x_data[columns_filter]], axis=1).astype('float32')
    x_test_2 = pd.concat([x_data_hog_test, x_data_test[columns_filter]], axis=1).astype('float32')
  else:
    x_train_2 = pd.DataFrame(x_data[columns_filter]).astype('float32')
    x_test_2 = pd.DataFrame(x_data_test[columns_filter]).astype('float32')

  scaler = StandardScaler()

  x_train_2[columns_filter] = scaler.fit_transform(x_train_2[columns_filter])
  x_test_2[columns_filter] = scaler.transform(x_test_2[columns_filter])

  total_epochs_per_repeat = 20

  for i in range(5):
    print("---- Repeat number " + str(i+1) + " ----")
    model_history = model_autoencoder.fit(x_train_2, epochs=total_epochs_per_repeat, batch_size=32, verbose=1, workers=4,)

    print("---- Calculating SVM - Repeat number " + str(i+1) + " ----")

    x_train = model_autoencoder.encoder(np.array(x_train_2))[2]
    x_test = model_autoencoder.encoder(np.array(x_test_2))[2]

    params = getBestParams(results, x_train.numpy(), x_data[target_val], with_pca=False, binary=binary, groups=x_data["video_name"])
    train_svm(results, x_train.numpy(), x_data[target_val], x_test, x_data_test[target_val], model_name, params, x_data["video_name"], weights, False, binary)

