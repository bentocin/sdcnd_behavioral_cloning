# TODO: Python generator for loading training data instead of storing in memory
# TODO: Put comments where needed and organize the code
# TODO: Appropriate model is implemented (filter sizes match, nonlinearities included, normalization, dropout)
# TODO: Train/validation/test splits are included
# TODO: Description on why specific learning rate or optimizer was chosen

# HINT: Use successive refinement of the model just like transfer learning and finetune with more data and lower learning rates
# HINT: Use the 50Hz simulator with fastest graphic quality and the lowest resolution
# HINT: Generate at least 40k images
# HINT: Copy the Nvidia pipeline
# HINT: Resize the input images
# HINT: Make use of the left and right images as well by adapting the steering value for them
# HINT: Around 5 epochs seem to be enough
# HINT: When starting out start by picking three images from the csv file one with left steering, one with right, and one straight
#       Train a model just with these three images and see if you can get it to predict them correctly

import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten, Dropout, Lambda
from keras.models import Sequential

def load_data(data_folder, correction):

  data = pd.read_csv(os.path.join(data_folder, "driving_log.csv"))

  # Get the file paths from csv and extract the filename only
  X = data[['center', 'left', 'right']].values
  # Get the steering values and adjust them for the left and right images
  y = data['steering'].values

  # Get indices of training and validation data
  train_ind, valid_ind = train_test_split(range(X.shape[0]), test_size=0.2, random_state=42)

  # For training use center, left, and right images
  X_train = np.array([fname.replace("\\", "/").split("/")[-1] for fname in X[train_ind].flatten(order="F")])
  y_train = np.hstack((y[train_ind], y[train_ind]+correction, y[train_ind]-correction))
  # For validation only use center images
  X_valid = np.array([fname.replace("\\", "/").split("/")[-1] for fname in X[valid_ind][:,0].flatten(order="F")])
  y_valid = y[valid_ind]

  return X_train, X_valid, y_train, y_valid

def read_img(file_name):
  # Read in image as BGR format
  img = cv2.imread(file_name)
  return img

def preprocess_img(img):
  # Remove the sky and the car
  crop = img[65:-25, :, :] 
  # Resize to fit for Nvidia model and change color space accordingly
  resize = cv2.resize(crop, (200, 66))
  color = cv2.cvtColor(resize, cv2.COLOR_BGR2YUV)
  return color

def flip_img(img, steering):
  # Flip image horizontally and adjust steering angle
  flip = cv2.flip(img, 1)
  angle = -steering
  return flip, angle

def data_generator(img_folder, X_data, y_data, batch_size=128, is_training=True):
  num_samples = len(X_data)
  while 1:
    # Shuffle the data
    X_data, y_data = shuffle(X_data, y_data)
    # Loop over the dataset with batch size step
    for offset in range(0, num_samples, batch_size):
      X_batch = X_data[offset:offset+batch_size]
      y_batch = y_data[offset:offset+batch_size]

      images = []
      angles = []

      # Loop over each sample in teh batch
      for fname, angle in zip(X_batch, y_batch):
        img = read_img(os.path.join(img_folder, fname))
        ang = angle
        # If training randomly augment the image by flipping it
        if is_training and np.random.rand() < 0.5:
          img, ang = flip_img(img, angle)
        # Preprocess data for the model
        img = preprocess_img(img)
        images.append(img)
        angles.append(ang)
      
      X = np.array(images)
      y = np.array(angles)

      yield shuffle(X, y)

def build_model():
  # Build the Nvidia model with small adaptions
  model = Sequential()
  model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(66,200,3)))
  model.add(Convolution2D(24, 5, 5, activation='relu', subsample=(2, 2)))
  model.add(Convolution2D(36, 5, 5, activation='relu', subsample=(2, 2)))
  model.add(Convolution2D(48, 5, 5, activation='relu', subsample=(2, 2)))
  model.add(Convolution2D(64, 3, 3, activation='relu'))
  model.add(Convolution2D(64, 3, 3, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Flatten())
  model.add(Dense(100, activation='relu'))
  model.add(Dense(50, activation='relu'))
  model.add(Dense(10, activation='relu'))
  model.add(Dense(1))

  return model

def train_model(model, X_train, X_valid, y_train, y_valid, batch_size, epochs, img_folder):
  # Create callbacks for saving checkpoints and early stopping
  checkpoint = ModelCheckpoint("model.h5", save_best_only=True)
  early_stopping = EarlyStopping(min_delta=0.1, patience=3)
  
  model.compile(loss='mse', optimizer="adam")
  # Number of batches to yield from generator for one epoch
  train_steps = (X_train.shape[0]//3//batch_size)*batch_size
  valid_steps = (X_valid.shape[0]//batch_size)*batch_size
  # Train the model
  model.fit_generator(data_generator(img_folder, X_train, y_train, batch_size, True), train_steps, max_q_size=1,
                      nb_epoch=epochs, validation_data=data_generator(img_folder, X_valid, y_valid, batch_size, False),
                      nb_val_samples=valid_steps, callbacks=[checkpoint, early_stopping])

def main():
  data_folder = os.path.join(os.path.curdir, "data")
  img_folder = os.path.join(data_folder, "IMG")
  batch_size = 128
  learning_rate = 0.001
  correction = 0.2
  epochs = 3

  X_train, X_valid, y_train, y_valid = load_data(data_folder, correction)
  model = build_model()
  train_model(model, X_train, X_valid, y_train, y_valid, batch_size, epochs, img_folder)

if __name__ == "__main__":
  main()