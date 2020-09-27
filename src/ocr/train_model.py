#!/usr/bin/env python3

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from model import model
from dataset import helper
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import build_montages
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--az", required=True, help="path to A-Z dataset")
ap.add_argument("-m", "--model", type=str, required=True, help="path to output trained handwriting recognition model")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output training history file")
args = vars(ap.parse_args())

# === STARTING PARAMS ===
# initialize the number of epochs to train for, initial learning rate and batch size
EPOCHS = 30
INIT_LR = 1e-2
BS = 6

# load the dataset
print("[INFO] loading datasets...")
data, labels = helper.load_dataset(args["az"])

# Our vectorized training labels
labels = to_categorical(labels)


# add a channel dimension to every image in the dataset 
data = np.expand_dims(data, axis=-1)

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# construct the image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=10,
	zoom_range=0.05,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.15,
	horizontal_flip=False,
	fill_mode="nearest")


# initialize and compile our deep neural network
print("[INFO] Compiling model...")
opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model = model.get_model()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


# train the network
print("[INFO] Training network...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS,
	verbose=1)

# define the list of label names
labelNames = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelNames = [l for l in labelNames]

# evaluate the network
# print("[INFO] Evaluating network...")
# predictions = model.predict(testX, batch_size=BS)

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"], save_format="h5")

# construct a plot that plots and saves the training history
print("[INFO] Generating loss plot")
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
