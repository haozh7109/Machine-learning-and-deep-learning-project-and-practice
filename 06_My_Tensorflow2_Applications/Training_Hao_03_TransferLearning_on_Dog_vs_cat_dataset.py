#!/usr/bin/env python
# coding: utf-8

# # Programming assignment

# ## Transfer learning

# ### Instructions
# 
# In this notebook, you will create a neural network model to classify images of cats and dogs, using transfer learning: you will use part of a pre-trained image classifier model (trained on ImageNet) as a feature extractor, and train additional new layers to perform the cats and dogs classification task.
# 
# Some code cells are provided you in the notebook. You should avoid editing provided code, and make sure to execute the cells in order to avoid unexpected errors. Some cells begin with the line: 
# 
# `#### GRADED CELL ####`
# 
# Don't move or edit this first line - this is what the automatic grader looks for to recognise graded cells. These cells require you to write your own code to complete them, and are automatically graded when you submit the notebook. Don't edit the function name or signature provided in these cells, otherwise the automatic grader might not function properly. Inside these graded cells, you can use any functions or classes that are imported below, but make sure you don't use any variables that are outside the scope of the function.
# 
# ### How to submit
# 
# Complete all the tasks you are asked for in the worksheet. When you have finished and are happy with your code, press the **Submit Assignment** button at the top of this notebook.
# 
# ### Let's get started!
# 
# We'll start running some imports, and loading the dataset. Do not edit the existing imports in the following cell. If you would like to make further Tensorflow imports, you should add them here.

# In[15]:


#### PACKAGE IMPORTS ####

# Run this cell first to import all required packages. Do not make any imports elsewhere in the notebook

import tensorflow as tf
from tensorflow.keras.models import  Sequential, Model, load_model
import numpy as np
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# If you would like to make further imports from Tensorflow, add them here
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,Dense,Flatten,Dropout


# <img src="data/Cats-Dogs-Rex.jpg" alt="Drawing" style="height: 450px;" align="center"/>
# 
# #### The Dogs vs Cats dataset
# 
# In this assignment, you will use the [Dogs vs Cats dataset](https://www.kaggle.com/c/dogs-vs-cats/data), which was used for a 2013 Kaggle competition. It consists of 25000 images containing either a cat or a dog. We will only use a subset of 600 images and labels. The dataset is a subset of a much larger dataset of 3 million photos that were originally used as a CAPTCHA (Completely Automated Public Turing test to tell Computers and Humans Apart), referred to as “Asirra” or Animal Species Image Recognition for Restricting Access.
# 
# * J. Elson, J. Douceur, J. Howell, and J. Saul. "Asirra: A CAPTCHA that Exploits Interest-Aligned Manual Image Categorization." Proceedings of 14th ACM Conference on Computer and Communications Security (CCS), October 2007.
# 
# Your goal is to train a classifier model using part of a pre-trained image classifier, using the principle of transfer learning.

# #### Load and preprocess the data

# In[2]:


images_train = np.load('data/images_train.npy') / 255.
images_valid = np.load('data/images_valid.npy') / 255.
images_test = np.load('data/images_test.npy') / 255.

labels_train = np.load('data/labels_train.npy')
labels_valid = np.load('data/labels_valid.npy')
labels_test = np.load('data/labels_test.npy')


# In[3]:


print("{} training data examples".format(images_train.shape[0]))
print("{} validation data examples".format(images_valid.shape[0]))
print("{} test data examples".format(images_test.shape[0]))


# #### Display sample images and labels from the training set

# In[4]:


# Display a few images and labels

class_names = np.array(['Dog', 'Cat'])

plt.figure(figsize=(15,10))
inx = np.random.choice(images_train.shape[0], 15, replace=False)
for n, i in enumerate(inx):
    ax = plt.subplot(3,5,n+1)
    plt.imshow(images_train[i])
    plt.title(class_names[labels_train[i]])
    plt.axis('off')


# #### Create a benchmark model
# 
# We will first train a CNN classifier model as a benchmark model before implementing the transfer learning approach. Using the functional API, build the benchmark model according to the following specifications:
# 
# * The model should use the `input_shape` in the function argument to set the shape in the Input layer.
# * The first and second hidden layers should be Conv2D layers with 32 filters, 3x3 kernel size and ReLU activation.
# * The third hidden layer should be a MaxPooling2D layer with a 2x2 window size.
# * The fourth and fifth hidden layers should be Conv2D layers with 64 filters, 3x3 kernel size and ReLU activation.
# * The sixth hidden layer should be a MaxPooling2D layer with a 2x2 window size.
# * The seventh and eighth hidden layers should be Conv2D layers with 128 filters, 3x3 kernel size and ReLU activation.
# * The ninth hidden layer should be a MaxPooling2D layer with a 2x2 window size.
# * This should be followed by a Flatten layer, and a Dense layer with 128 units and ReLU activation
# * The final layer should be a Dense layer with a single neuron and sigmoid activation.
# * All of the Conv2D layers should use `'SAME'` padding.
# 
# In total, the network should have 13 layers (including the `Input` layer).
# 
# The model should then be compiled with the RMSProp optimiser with learning rate 0.001, binary cross entropy loss and and binary accuracy metric.

# In[5]:


#### GRADED CELL ####

# Complete the following function. 
# Make sure to not change the function name or arguments.

def get_benchmark_model(input_shape):
    """
    This function should build and compile a CNN model according to the above specification,
    using the functional API. The function takes input_shape as an argument, which should be
    used to specify the shape in the Input layer.
    Your function should return the model.
    """
    
    # define the model 
    inputs = Input(shape=input_shape)
    h      = Conv2D(32,(3,3),activation='relu',padding='SAME')(inputs)
    h      = Conv2D(32,(3,3),activation='relu',padding='SAME')(h)
    h      = MaxPooling2D((2,2))(h)
    h      = Conv2D(64,(3,3),activation='relu',padding='SAME')(h)
    h      = Conv2D(64,(3,3),activation='relu',padding='SAME')(h)
    h      = MaxPooling2D((2,2))(h)
    h      = Conv2D(128,(3,3),activation='relu',padding='SAME')(h)
    h      = Conv2D(128,(3,3),activation='relu',padding='SAME')(h)
    h      = MaxPooling2D((2,2))(h)
    h      = Flatten()(h)
    h      = Dense(128,activation='relu')(h)
    outputs = Dense(1,activation='sigmoid')(h)
    
    model = Model(inputs=inputs,outputs=outputs)

    #compile the model
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),loss="binary_crossentropy",metrics=['accuracy'])
    
    return model


# In[6]:


# Build and compile the benchmark model, and display the model summary

benchmark_model = get_benchmark_model(images_train[0].shape)
benchmark_model.summary()


# #### Train the CNN benchmark model
# 
# We will train the benchmark CNN model using an `EarlyStopping` callback. Feel free to increase the training time if you wish.

# In[7]:


# Fit the benchmark model and save its training history

earlystopping = tf.keras.callbacks.EarlyStopping(patience=2)
history_benchmark = benchmark_model.fit(images_train, labels_train, epochs=10, batch_size=32,
                                        validation_data=(images_valid, labels_valid), 
                                        callbacks=[earlystopping])


# #### Plot the learning curves

# In[8]:


# Run this cell to plot accuracy vs epoch and loss vs epoch

plt.figure(figsize=(15,5))
plt.subplot(121)
try:
    plt.plot(history_benchmark.history['accuracy'])
    plt.plot(history_benchmark.history['val_accuracy'])
except KeyError:
    plt.plot(history_benchmark.history['acc'])
    plt.plot(history_benchmark.history['val_acc'])
plt.title('Accuracy vs. epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')

plt.subplot(122)
plt.plot(history_benchmark.history['loss'])
plt.plot(history_benchmark.history['val_loss'])
plt.title('Loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show() 


# #### Evaluate the benchmark model

# In[9]:


# Evaluate the benchmark model on the test set

benchmark_test_loss, benchmark_test_acc = benchmark_model.evaluate(images_test, labels_test, verbose=0)
print("Test loss: {}".format(benchmark_test_loss))
print("Test accuracy: {}".format(benchmark_test_acc))


# #### Load the pretrained image classifier model
# 
# You will now begin to build our image classifier using transfer learning.
# You will use the pre-trained MobileNet V2 model, available to download from [Keras Applications](https://keras.io/applications/#mobilenetv2). However, we have already downloaded the pretrained model for you, and it is available at the location `./models/MobileNetV2.h5`.

# In[11]:


#### GRADED CELL ####

# Complete the following function. 
# Make sure to not change the function name or arguments.

def load_pretrained_MobileNetV2(path):
    """
    This function takes a path as an argument, and uses it to 
    load the full MobileNetV2 pretrained model from the path.
    Your function should return the loaded model.
    """
    model = load_model(path)
    return model
    


# In[12]:


# Call the function loading the pretrained model and display its summary

base_model = load_pretrained_MobileNetV2('models/MobileNetV2.h5')
base_model.summary()


# #### Use the pre-trained model as a feature extractor
# 
# You will remove the final layer of the network and replace it with new, untrained classifier layers for our task. You will first create a new model that has the same input tensor as the MobileNetV2 model, and uses the output tensor from the layer with name `global_average_pooling2d_6` as the model output.

# In[13]:


#### GRADED CELL ####

# Complete the following function. 
# Make sure to not change the function name or arguments.

def remove_head(pretrained_model):
    """
    This function should create and return a new model, using the input and output 
    tensors as specified above. 
    Use the 'get_layer' method to access the correct layer of the pre-trained model.
    """
    # derive the input and designated layer's output of the pretrained model
    inputs  = pretrained_model.input
    outputs = pretrained_model.get_layer("global_average_pooling2d_6").output
    
    # define a new model
    model   = Model(inputs=inputs,outputs = outputs)
    
    return model


# In[14]:


# Call the function removing the classification head and display the summary

feature_extractor = remove_head(base_model)
feature_extractor.summary()


# You can now construct new final classifier layers for your model. Using the Sequential API, create a new model according to the following specifications:
# 
# * The new model should begin with the feature extractor model.
# * This should then be followed with a new dense layer with 32 units and ReLU activation function.
# * This should be followed by a dropout layer with a rate of 0.5.
# * Finally, this should be followed by a Dense layer with a single neuron and a sigmoid activation function.
# 
# In total, the network should be composed of the pretrained base model plus 3 layers.

# In[16]:


#### GRADED CELL ####

# Complete the following function. 
# Make sure to not change the function name or arguments.

def add_new_classifier_head(feature_extractor_model):
    """
    This function takes the feature extractor model as an argument, and should create
    and return a new model according to the above specification.
    """
    
    model = Sequential([
        feature_extractor_model,
        Dense(32,activation='relu'),
        Dropout(0.5),
        Dense(1,activation='sigmoid')
    ])
    
    return model
    


# In[17]:


# Call the function adding a new classification head and display the summary

new_model = add_new_classifier_head(feature_extractor)
new_model.summary()


# #### Freeze the weights of the pretrained model

# You will now need to freeze the weights of the pre-trained feature extractor, so that only the weights of the new layers you have added will change during the training. 
# 
# You should then compile your model as before: use the RMSProp optimiser with learning rate 0.001, binary cross entropy loss and and binary accuracy metric.

# In[18]:


#### GRADED CELL ####

# Complete the following function. 
# Make sure to not change the function name or arguments.

def freeze_pretrained_weights(model):
    """
    This function should freeze the weights of the pretrained base model.
    Your function should return the model with frozen weights.
    """
    #freeze the pretrained model weights
    model.layers[0].trainable = False
    
    # compile the model
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),loss="binary_crossentropy",metrics=['accuracy'])
    
    return model
    


# In[19]:


# Call the function freezing the pretrained weights and display the summary

frozen_new_model = freeze_pretrained_weights(new_model)
frozen_new_model.summary()


# #### Train the model
# 
# You are now ready to train the new model on the dogs vs cats data subset. We will use an `EarlyStopping` callback with patience set to 2 epochs, as before. Feel free to increase the training time if you wish.

# In[20]:


# Train the model and save its training history

earlystopping = tf.keras.callbacks.EarlyStopping(patience=2)
history_frozen_new_model = frozen_new_model.fit(images_train, labels_train, epochs=10, batch_size=32,
                                                validation_data=(images_valid, labels_valid), 
                                                callbacks=[earlystopping])


# #### Plot the learning curves

# In[21]:


# Run this cell to plot accuracy vs epoch and loss vs epoch

plt.figure(figsize=(15,5))
plt.subplot(121)
try:
    plt.plot(history_frozen_new_model.history['accuracy'])
    plt.plot(history_frozen_new_model.history['val_accuracy'])
except KeyError:
    plt.plot(history_frozen_new_model.history['acc'])
    plt.plot(history_frozen_new_model.history['val_acc'])
plt.title('Accuracy vs. epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')

plt.subplot(122)
plt.plot(history_frozen_new_model.history['loss'])
plt.plot(history_frozen_new_model.history['val_loss'])
plt.title('Loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show() 


# #### Evaluate the new model

# In[22]:


# Evaluate the benchmark model on the test set

new_model_test_loss, new_model_test_acc = frozen_new_model.evaluate(images_test, labels_test, verbose=0)
print("Test loss: {}".format(new_model_test_loss))
print("Test accuracy: {}".format(new_model_test_acc))


# #### Compare both models
# 
# Finally, we will look at the comparison of training, validation and test metrics between the benchmark and transfer learning model.

# In[23]:


# Gather the benchmark and new model metrics

benchmark_train_loss = history_benchmark.history['loss'][-1]
benchmark_valid_loss = history_benchmark.history['val_loss'][-1]

try:
    benchmark_train_acc = history_benchmark.history['acc'][-1]
    benchmark_valid_acc = history_benchmark.history['val_acc'][-1]
except KeyError:
    benchmark_train_acc = history_benchmark.history['accuracy'][-1]
    benchmark_valid_acc = history_benchmark.history['val_accuracy'][-1]

new_model_train_loss = history_frozen_new_model.history['loss'][-1]
new_model_valid_loss = history_frozen_new_model.history['val_loss'][-1]

try:
    new_model_train_acc = history_frozen_new_model.history['acc'][-1]
    new_model_valid_acc = history_frozen_new_model.history['val_acc'][-1]
except KeyError:
    new_model_train_acc = history_frozen_new_model.history['accuracy'][-1]
    new_model_valid_acc = history_frozen_new_model.history['val_accuracy'][-1]


# In[24]:


# Compile the metrics into a pandas DataFrame and display the table

comparison_table = pd.DataFrame([['Training loss', benchmark_train_loss, new_model_train_loss],
                                ['Training accuracy', benchmark_train_acc, new_model_train_acc],
                                ['Validation loss', benchmark_valid_loss, new_model_valid_loss],
                                ['Validation accuracy', benchmark_valid_acc, new_model_valid_acc],
                                ['Test loss', benchmark_test_loss, new_model_test_loss],
                                ['Test accuracy', benchmark_test_acc, new_model_test_acc]],
                               columns=['Metric', 'Benchmark CNN', 'Transfer learning CNN'])
comparison_table.index=['']*6
comparison_table


# In[25]:


# Plot confusion matrices for benchmark and transfer learning models

plt.figure(figsize=(15, 5))

preds = benchmark_model.predict(images_test)
preds = (preds >= 0.5).astype(np.int32)
cm = confusion_matrix(labels_test, preds)
df_cm = pd.DataFrame(cm, index=['Dog', 'Cat'], columns=['Dog', 'Cat'])
plt.subplot(121)
plt.title("Confusion matrix for benchmark model\n")
sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")
plt.ylabel("Predicted")
plt.xlabel("Actual")

preds = frozen_new_model.predict(images_test)
preds = (preds >= 0.5).astype(np.int32)
cm = confusion_matrix(labels_test, preds)
df_cm = pd.DataFrame(cm, index=['Dog', 'Cat'], columns=['Dog', 'Cat'])
plt.subplot(122)
plt.title("Confusion matrix for transfer learning model\n")
sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")
plt.ylabel("Predicted")
plt.xlabel("Actual")
plt.show()


# Congratulations for completing this programming assignment! In the next week of the course we will learn how to develop an effective data pipeline.
