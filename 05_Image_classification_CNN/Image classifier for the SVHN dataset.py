# -*- coding: utf-8 -*-
"""Capstone Project for applying Deep CNN for image Classification

Code by Hao, Zhao, Aug, 2020.


# Capstone Project
## Image classifier for the SVHN dataset
### Instructions

In this notebook, you will create a neural network that classifies real-world images digits. You will use concepts from throughout this course in building, training, testing, validating and saving your Tensorflow classifier model.

This project is peer-assessed. Within this notebook you will find instructions in each section for how to complete the project. Pay close attention to the instructions as the peer review will be carried out according to a grading rubric that checks key parts of the project instructions. Feel free to add extra cells into the notebook as required.

### How to submit

When you have completed the Capstone project notebook, you will submit a pdf of the notebook for peer review. First ensure that the notebook has been fully executed from beginning to end, and all of the cell outputs are visible. This is important, as the grading rubric depends on the reviewer being able to view the outputs of your notebook. Save the notebook as a pdf (you could download the notebook with File -> Download .ipynb, open the notebook locally, and then File -> Download as -> PDF via LaTeX), and then submit this pdf for review.

### Let's get started!

We'll start by running some imports, and loading the dataset. For this project you are free to make further imports throughout the notebook as you wish.

"""

import tensorflow as tf
from scipy.io import loadmat

"""For the capstone project, you will use the [SVHN dataset](http://ufldl.stanford.edu/housenumbers/). This is an image dataset of over 600,000 digit images in all, and is a harder dataset than MNIST as the numbers appear in the context of natural scene images. SVHN is obtained from house numbers in Google Street View images.

* Y. Netzer, T. Wang, A. Coates, A. Bissacco, B. Wu and A. Y. Ng. "Reading Digits in Natural Images with Unsupervised Feature Learning". NIPS Workshop on Deep Learning and Unsupervised Feature Learning, 2011.

The train and test datasets required for this project can be downloaded from [here](http://ufldl.stanford.edu/housenumbers/train.tar.gz) and [here](http://ufldl.stanford.edu/housenumbers/test.tar.gz). Once unzipped, you will have two files: `train_32x32.mat` and `test_32x32.mat`. You should store these files in Drive for use in this Colab notebook.

Your goal is to develop an end-to-end workflow for building, training, validating, evaluating and saving a neural network that classifies a real-world image into one of ten classes.
"""

# Run this cell to connect to your Drive folder

# from google.colab import drive
# drive.mount('/content/gdrive')

# Load the dataset from your Drive folder

train = loadmat('test_data_folder/SVHN/train_32x32.mat')
test = loadmat('test_data_folder/SVHN/test_32x32.mat')

"""Both `train` and `test` are dictionaries with keys `X` and `y` for the input images and labels respectively.

## 1. Inspect and preprocess the dataset
* Extract the training and testing images and labels separately from the train and test dictionaries loaded for you.
* Select a random sample of images and corresponding labels from the dataset (at least 10), and display them in a figure.
* Convert the training and test images to grayscale by taking the average across all colour channels for each pixel. _Hint: retain the channel dimension, which will now have size 1._
* Select a random sample of the grayscale images and corresponding labels from the dataset (at least 10), and display them in a figure.
"""

#### PACKAGE IMPORTS ####

from numpy.random import seed
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from sklearn import datasets, model_selection 
# get_ipython().run_line_magic('matplotlib', 'inline')

# import tensorflow packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,BatchNormalization, Dropout, Conv2D, MaxPool2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

# (1) extract the training and testing iamges and labels

train_images = train["X"]
train_labels = train["y"]
test_images  = test["X"]
test_labels  = test["y"]

# convert the image from original int8 to float32 precision
train_images = train_images.astype("float32")
train_labels = train_labels.astype("float32")
test_images  = test_images.astype("float32")
test_labels  = test_labels.astype("float32")

# swap the array order of train and test images from [x,y,chan,sample] to [sample,x,y,chan]
train_images = train_images.transpose(3,0,1,2)
test_images  = test_images.transpose(3,0,1,2)

# Convert targets to a one-hot encoding

train_targets = train_labels
test_targets  = test_labels

# revert the label 10 to 0 to avoid the problem in loss calculation
train_targets[train_targets==10] = 0
test_targets[test_targets==10]   = 0 

train_targets = tf.keras.utils.to_categorical(np.array(train_labels),num_classes=10)
test_targets = tf.keras.utils.to_categorical(np.array(test_labels),num_classes=10)

  
# (2) Select a random sample of images and corresponding labels from the dataset (at least 10), and display them in a figure.

# display a simgle image
n = 100
img = train_images[n,:,:,0]

fig1 = plt.figure(1)
plt.imshow(img)
plt.title("label = {}".format(train_labels[n]))
fig1.show()


# plot 10 random selected images
fig2, ax = plt.subplots(1, 10, figsize=(10, 1))
for i in range(10):
    n = random.randint(10,train_labels.shape[0])
    ax[i].set_axis_off()
    ax[i].imshow(train_images[n,:,:,0])
    ax[i].set_title("label = {}".format(train_labels[n]))

fig2.show()
 
# (3) Convert the training and test images to grayscale by taking the average across all colour channels for each pixel.

train_images_greyscale = train_images.mean(axis=3)
test_images_greyscale  = test_images.mean(axis=3)


# normalize the train and test set 
train_images_greyscale = train_images_greyscale / 255
test_images_greyscale  = test_images_greyscale / 255

# retain the channel dimmension
train_images_greyscale = train_images_greyscale[...,np.newaxis]
test_images_greyscale  = test_images_greyscale[...,np.newaxis]


# (4) Select a random sample of the grayscale images and corresponding labels from the dataset (at least 10), and display them in a figure.

# display a simgle image
n = 100
img = train_images_greyscale[n,:,:,0]

fig3 = plt.figure(3)
plt.imshow(img,cmap='gray')
plt.title("label = {}".format(train_labels[n]))
fig3.show()


# plot 10 random selected images
fig4, ax = plt.subplots(1, 10, figsize=(10, 1))
for i in range(10):
    n = random.randint(10,train_labels.shape[0])
    ax[i].set_axis_off()
    ax[i].imshow(train_images_greyscale[n,:,:,0],cmap='gray')
    ax[i].set_title("label = {}".format(train_labels[n]))
fig4.show()



"""## 2. MLP neural network classifier
* Build an MLP classifier model using the Sequential API. Your model should use only Flatten and Dense layers, with the final layer having a 10-way softmax output. 
* You should design and build the model yourself. Feel free to experiment with different MLP architectures. _Hint: to achieve a reasonable accuracy you won't need to use more than 4 or 5 layers._
* Print out the model summary (using the summary() method)
* Compile and train the model (we recommend a maximum of 30 epochs), making use of both training and validation sets during the training run. 
* Your model should track at least one appropriate metric, and use at least two callbacks during training, one of which should be a ModelCheckpoint callback.
* As a guide, you should aim to achieve a final categorical cross entropy training loss of less than 1.0 (the validation loss might be higher).
* Plot the learning curves for loss vs epoch and accuracy vs epoch for both training and validation sets.
* Compute and display the loss and accuracy of the trained model on the test set.
"""


# define the MLP neural network classifier

# (1) define the MLP model by a function
def get_MLP_model():
    model = tf.keras.models.Sequential([
      Flatten(input_shape=(32, 32, 1)),
      Dense(128,activation='relu'),
      Dense(128,activation='relu'),
      Dense(128,activation='relu'),
      Dense(128,activation='relu'),
      Dense(10,activation='softmax')
      ])
 
    return model 



#(2) define the function to compile the model
    
def get_model_compile_MLP(model):
    
    optim = 'adam'
    los   = tf.keras.losses.CategoricalCrossentropy()
    mec   = ['accuracy']
    
    model.compile(optimizer=optim,
              loss=los,
              metrics=mec)

#(3) define the call back 
    
def get_callbacks_MLP():
    
    # define early stop call back.
    early_stopping=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,mode='min')
    
    # define model check point call back for saving models in training
    checkpoint_path = 'MLP_model_checkpoints'
    checkpoint      = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=False,frequency='epoch',verbose=1)
    
    return (early_stopping, checkpoint)
    

#(4) get and compile the MLP model 
    
model = get_MLP_model()
get_model_compile_MLP(model)


print(model.loss)
print(model.optimizer)
print(model.metrics)

#(5) model training

# get the call backs
early_stopping, checkpoint = get_callbacks_MLP()


start_time = time.time()

# with tf.device('/CPU:0'):
    
history = model.fit(train_images_greyscale,train_targets,epochs=30,batch_size=128,validation_split=0.2,callbacks=[early_stopping, checkpoint])

end_time = time.time()

print("======== The training is done in {} second =========".format(end_time-start_time))


#(6) plot training history

df = pd.DataFrame(history.history)
df.head(30)


# Plot the training and validation loss
fig5 = plt.figure(5)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
fig5.show()

# Plot the training and validation accuracy
fig6 = plt.figure(6)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy vs. epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
fig6.show()

#(7) plot test accuracy

test_loss, test_acc = model.evaluate(test_images_greyscale,test_targets, verbose=0)
print('===========================================================')
print('MLP model derived accuracy: {acc:0.3f}'.format(acc=test_acc))
print('===========================================================')


"""## 3. CNN neural network classifier
* Build a CNN classifier model using the Sequential API. Your model should use the Conv2D, MaxPool2D, BatchNormalization, Flatten, Dense and Dropout layers. The final layer should again have a 10-way softmax output. 
* You should design and build the model yourself. Feel free to experiment with different CNN architectures. _Hint: to achieve a reasonable accuracy you won't need to use more than 2 or 3 convolutional layers and 2 fully connected layers.)_
* The CNN model should use fewer trainable parameters than your MLP model.
* Compile and train the model (we recommend a maximum of 30 epochs), making use of both training and validation sets during the training run.
* Your model should track at least one appropriate metric, and use at least two callbacks during training, one of which should be a ModelCheckpoint callback.
* You should aim to beat the MLP model performance with fewer parameters!
* Plot the learning curves for loss vs epoch and accuracy vs epoch for both training and validation sets.
* Compute and display the loss and accuracy of the trained model on the test set.
"""

# (1) define the CNN model by a function
def get_CNN_model():
    model = tf.keras.models.Sequential([
        Conv2D(32,(3,3),padding='same',activation='relu',input_shape= (32,32,1)),
        MaxPool2D((2,2)),
        Dropout(0.2),
        Conv2D(64,(3,3),activation='relu'),
        MaxPool2D((2,2)),
        BatchNormalization(),
        Conv2D(32,(3,3),activation='relu'),
        Dropout(0.2),
        Flatten(),
        Dense(64,activation='relu'),
        tf.keras.layers.Dense(10,activation='softmax')
      ])
 
    return model 


#(2) define the function to compile the model
    
def get_model_compile_CNN(model):
    
    optim = 'adam'
    los   = tf.keras.losses.CategoricalCrossentropy()
    mec   = ['accuracy']
    
    model.compile(optimizer=optim,
              loss=los,
              metrics=mec)

#(3) define the call back 
    
def get_callbacks_CNN():
    
    # define early stop call back.
    early_stopping=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,mode='min')
    
    # define model check point call back for saving models in training
    checkpoint_path = 'CNN_model_checkpoints'
    checkpoint      = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=False,frequency='epoch',verbose=1)
    
    return (early_stopping, checkpoint)
    

#(4) get and compile the MLP model 
    
model = get_CNN_model()
get_model_compile_CNN(model)

print(model.summary())
print(model.loss)
print(model.optimizer)
print(model.metrics)


#(5) model training

# get the call backs
early_stopping, checkpoint = get_callbacks_CNN()


start_time = time.time()

with tf.device('/CPU:0'):
    
    history = model.fit(train_images_greyscale,train_targets,epochs=30,batch_size=128,validation_split=0.2,callbacks=[early_stopping, checkpoint])

end_time = time.time()

print("======== The training is done in {} second =========".format(end_time-start_time))


#(6) plot training history

df = pd.DataFrame(history.history)
df.head(30)


# Plot the training and validation loss
fig7 = plt.figure(7)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
fig7.show()

# Plot the training and validation accuracy
fig8 = plt.figure(8)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy vs. epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
fig8.show()

#(7) plot test accuracy

test_loss, test_acc = model.evaluate(test_images_greyscale,test_targets, verbose=0)
print('===========================================================')
print('CNN model derived accuracy: {acc:0.3f}'.format(acc=test_acc))
print('===========================================================')


"""## 4. Get model predictions
* Load the best weights for the MLP and CNN models that you saved during the training run.
* Randomly select 5 images and corresponding labels from the test set and display the images with their labels.
* Alongside the image and label, show each model’s predictive distribution as a bar chart, and the final model prediction given by the label with maximum probability.
"""

# (1) reload the previously saved model 

from tensorflow.keras.models import load_model

model_MLP = load_model('MLP_model_checkpoints')
model_CNN = load_model('CNN_model_checkpoints')



#(2) check the model derived accuracy on the loaded model again

test_loss, test_acc = model_MLP.evaluate(test_images_greyscale,test_targets, verbose=0)
print('MLP model derived accuracy: {acc:0.3f}'.format(acc=test_acc))


test_loss, test_acc = model_CNN.evaluate(test_images_greyscale,test_targets, verbose=0)
print('CNN model derived accuracy: {acc:0.3f}'.format(acc=test_acc))


#(3) Randomly select 5 images and corresponding labels from the test set and display the images with their labels.

# generate 5 random test images
num_test_images = test_images_greyscale.shape[0]

random_inx = np.random.choice(num_test_images, 5)
random_test_images = test_images_greyscale[random_inx, ...]
random_test_labels = test_labels[random_inx, ...]

# get the model predictions 
predictions_MLP = model_MLP.predict(random_test_images)
predictions_CNN = model_CNN.predict(random_test_images)

# plot the MLP model labels VS. predictions

fig, axes = plt.subplots(5, 2, figsize=(16, 12))
fig.subplots_adjust(hspace=0.4, wspace=-0.2)

for i, (prediction, image, label) in enumerate(zip(predictions_MLP, random_test_images, random_test_labels)):
    axes[i, 0].imshow(np.squeeze(image))
    axes[i, 0].get_xaxis().set_visible(False)
    axes[i, 0].get_yaxis().set_visible(False)
    axes[i, 0].text(10., -1.5, f'Digit {label}')
    axes[i, 1].bar(np.arange(len(prediction)), prediction)
    axes[i, 1].set_xticks(np.arange(len(prediction)))
    axes[i, 1].set_title(f"Categorical distribution. MLP Model prediction: {np.argmax(prediction)}")
    
plt.show()

# plot the CNN model labels VS. predictions

fig, axes = plt.subplots(5, 2, figsize=(16, 12))
fig.subplots_adjust(hspace=0.4, wspace=-0.2)

for i, (prediction, image, label) in enumerate(zip(predictions_CNN, random_test_images, random_test_labels)):
    axes[i, 0].imshow(np.squeeze(image))
    axes[i, 0].get_xaxis().set_visible(False)
    axes[i, 0].get_yaxis().set_visible(False)
    axes[i, 0].text(10., -1.5, f'Digit {label}')
    axes[i, 1].bar(np.arange(len(prediction)), prediction)
    axes[i, 1].set_xticks(np.arange(len(prediction)))
    axes[i, 1].set_title(f"Categorical distribution. CNN Model prediction: {np.argmax(prediction)}")
    
plt.show()

