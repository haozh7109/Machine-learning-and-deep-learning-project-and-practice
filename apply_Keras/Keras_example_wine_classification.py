# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 13:18:33 2018

@author: haozh
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



## ----- part-1: load the red wine and white wine csv data by pandas -----------------------------------------------------------------------------------

white = pd.read_csv("C:\\Users\\haozh\\Documents\\GitHub\\Machine_Learning\\Keras_for_classification\\winequality-white.csv", sep=';')
red = pd.read_csv("C:\\Users\\haozh\\Documents\\GitHub\\Machine_Learning\\Keras_for_classification\\winequality-red.csv", sep=';')



## ----- part-2: QC the loaded data by pandas ----------------------------------------------------------------------------------------------------------
# Print info on white wine
print(white.info())

# Print info on red wine
print(red.info())

# Last rows of `white`
white.tail()

# Take a sample of 5 rows of `red`
red.sample(5)

# Describe `white`
white.describe()

# Double check for null values in `red`
pd.isnull(red)


# make a QC plot 
fig, ax = plt.subplots(1, 2)

ax[0].hist(red.alcohol, 10, facecolor='red', alpha=0.5, label="Red wine")
ax[1].hist(white.alcohol, 10, facecolor='white', ec="black", lw=0.5, alpha=0.5, label="White wine")

#fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=1)
ax[0].set_title("Red Wine")
ax[0].set_ylim([0, 1000])
ax[0].set_xlabel("Alcohol in % Vol")
ax[0].set_ylabel("Frequency")
ax[1].set_title("White Wine")
ax[1].set_xlabel("Alcohol in % Vol")
ax[1].set_ylabel("Frequency")
#ax[0].legend(loc='best')
#ax[1].legend(loc='best')
fig.suptitle("Distribution of Alcohol in % Vol")
plt.show()


# make a QC plot with sulphates vs quality 
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].scatter(red['quality'], red["sulphates"], color="red")
ax[1].scatter(white['quality'], white['sulphates'], color="white", edgecolors="black", lw=0.5)

ax[0].set_title("Red Wine")
ax[1].set_title("White Wine")
ax[0].set_xlabel("Quality")
ax[1].set_xlabel("Quality")
ax[0].set_ylabel("Sulphates")
ax[1].set_ylabel("Sulphates")
ax[0].set_xlim([0,10])
ax[1].set_xlim([0,10])
ax[0].set_ylim([0,2.5])
ax[1].set_ylim([0,2.5])
fig.subplots_adjust(wspace=0.5)
fig.suptitle("Wine Quality by Amount of Sulphates")

plt.show()


## ----- part-3: Data Preparation for Neural Netwrok Training ------------------------------------------------------------------------------------------

#----------------------------------Merge the two dataset ----------------------
# Add `type` column to `red` with value 1
red['type'] = 1

# Add `type` column to `white` with value 0
white['type'] = 0

# Append `white` to `red`
wines = red.append(white, ignore_index=True)


# plot the correlation matrix 
corr = wines.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
#sns.plt.show()

#----------------------------------Split data into training and test ----------

# Import `train_test_split` from `sklearn.model_selection`
from sklearn.model_selection import train_test_split

# Specify the data 
X=wines.ix[:,0:11]

# Specify the target labels and flatten the array 
y=np.ravel(wines.type)

# Split the data up in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


#----------------------------------Standardlize the training and test data ----
# Import `StandardScaler` from `sklearn.preprocessing`
from sklearn.preprocessing import StandardScaler

# Define the scaler 
scaler = StandardScaler().fit(X_train)

# Scale the train set
X_train = scaler.transform(X_train)

# Scale the test set
X_test = scaler.transform(X_test)


## ----- part-4: Build ANN network, apply training and prediction ------------------------------------------------------------------------------------------

# Import `Sequential` from `keras.models`
from keras.models import Sequential

# Import `Dense` from `keras.layers`
from keras.layers import Dense


#----------------------------------Build the ANN network ----------------------
# Initialize the constructor
model = Sequential()

# Add an input layer 
model.add(Dense(12, activation='relu', input_shape=(11,)))

# Add one hidden layer 
model.add(Dense(8, activation='relu'))

# Add an output layer 
model.add(Dense(1, activation='sigmoid'))


#----------------------------------Display the model setup ---------------------
# Model output shape
model.output_shape

# Model summary
model.summary()

# Model config
model.get_config()

# List all weight tensors 
model.get_weights()

#---------------------------compile the model and train with the data----------
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
                   
model.fit(X_train, y_train,epochs=20, batch_size=1, verbose=1)

#---------------------------make the prediction based on derived model---------
y_pred = model.predict(X_test)

y_pred = y_pred.astype(int)

print(y_pred[:5])
print(y_test[:5])

score = model.evaluate(X_test, y_test,verbose=1)

#--------------------------- Further evaluate the model ---------
# Import the modules from `sklearn.metrics`
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

# Confusion matrix
confusion_matrix(y_test, y_pred)

# Precision 
precision_score(y_test, y_pred)

# Recall
recall_score(y_test, y_pred)

# F1 score
f1_score(y_test,y_pred)

# Cohen's kappa
cohen_kappa_score(y_test, y_pred)

