# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 21:54:03 2020

@author: zhaoh
"""

import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import IPython.display as ipd
import os 
import time
import tensorflow as tf
import soundfile as sf

from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Dense,BatchNormalization,Dropout,Conv1D,SimpleRNN,AveragePooling1D,Flatten,Conv2D, MaxPool2D, Softmax
from random import random, seed
from sklearn.model_selection import train_test_split

from numpy import savetxt,loadtxt,savez_compressed,load
#-----------------------------------------------------------------------------
#  (1) define global variables and predefined methods 
#-----------------------------------------------------------------------------

# Parent Directory path 
parent_dir = "test_data_folder/audio_datasets/commonvoice/train/"

# new data directory 
data_directory = "Test_Speech_enhancement_data/"
  
# generate the folders
data_path = os.path.join(parent_dir, data_directory) 

try: 
    os.mkdir(data_path) 
    
except OSError as error:  
    print(error)

# set parameters for spectogram generation and data loading
n_fft              = 256
hop_length         = 64
sample_rate_signal = 22050
qc_file_id         = 1500
segment_len = 8
    
#-----------------------------------------------------------------------------
#  (2) Generate training and testing data
#-----------------------------------------------------------------------------

# load the meta data table
tb = pd.read_csv("test_data_folder/audio_datasets/commonvoice/train/train.tsv", sep='\t')


# generate dataset for training and testing

x = []
y = []

for file_index in np.arange(len(tb.path) // 2):
    # for file_index in np.arange(100):

    # load the clean and noisy signal's spectrogram
    clean_spec = load(data_path + tb.path[file_index] + '_clean_spec.npz')['arr_0']
    noisy_spec = load(data_path + tb.path[file_index] + '_noisy_spec.npz')['arr_0']

    # generate input (x) with dimension 129 * 8 , and tartget (y) with dimension 1 * 8
    segments_number = clean_spec.shape[1] - segment_len + 1

    for segment_index in np.arange(segment_len - 1, clean_spec.shape[1]):
        # extract the vector elements for the data
        x_element = noisy_spec[:, segment_index - segment_len + 1: segment_index + 1]
        y_element = clean_spec[:, segment_index]

        # save the data into the list
        x.append(x_element)
        y.append(y_element)
    
# convert data back to the array 
input_data  = np.array(x)
target_data = np.array(y)

#-----------------------------------------------------------------------------
#  (2) split data into training and test set
#-----------------------------------------------------------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(input_data, target_data, test_size=0.2, random_state=42)


# normalization the input and target of training set
X_train_mean = np.mean(X_train[:])
X_train_std  = np.std(X_train[:])
X_train      = (X_train - X_train_mean)/X_train_std
X_test       = (X_test  - X_train_mean)/X_train_std

y_train_mean = np.mean(y_train[:])
y_train_std  = np.std(y_train[:])
y_train      = (y_train - y_train_mean)/y_train_std
y_test       = (y_test  - y_train_mean)/y_train_std

# expand the dimension for 2D CNN
X_train = X_train[..., np.newaxis]
y_train = y_train[..., np.newaxis]
X_test  = X_test[..., np.newaxis]
y_test  = y_test[..., np.newaxis]
#-----------------------------------------------------------------------------
#  (3) Build the convolutional neural network for training 
#-----------------------------------------------------------------------------

CNN_model =  Sequential([Conv2D(18,(9,8),strides=(1,100), padding='SAME', activation='relu', input_shape=(129,8,1)),
                         BatchNormalization(),
                         
                         Conv2D(30,(5,1),strides=(1,100), padding='SAME', activation='relu'),
                         BatchNormalization(),
                         Conv2D(8,(9,1),strides=(1,100), padding='SAME', activation='relu'),
                         BatchNormalization(),                         
                         Conv2D(18,(9,1),strides=(1,100), padding='SAME', activation='relu'),
                         BatchNormalization(),   
                         
                         Conv2D(30,(5,1),strides=(1,100), padding='SAME', activation='relu'),
                         BatchNormalization(),
                         Conv2D(8,(9,1),strides=(1,100), padding='SAME', activation='relu'),
                         BatchNormalization(),                         
                         Conv2D(18,(9,1),strides=(1,100), padding='SAME', activation='relu'),
                         BatchNormalization(),   
                         
                         Conv2D(30,(5,1),strides=(1,100), padding='SAME', activation='relu'),
                         BatchNormalization(),
                         Conv2D(8,(9,1),strides=(1,100), padding='SAME', activation='relu'),
                         BatchNormalization(),                         
                         Conv2D(18,(9,1),strides=(1,100), padding='SAME', activation='relu'),
                         BatchNormalization(),   
                         
                         Conv2D(30,(5,1),strides=(1,100), padding='SAME', activation='relu'),
                         BatchNormalization(),
                         Conv2D(8,(9,1),strides=(1,100), padding='SAME', activation='relu'),
                         BatchNormalization(),                         
                         Conv2D(18,(9,1),strides=(1,100), padding='SAME', activation='relu'),
                         BatchNormalization(),   
                         
                         Conv2D(30,(5,1),strides=(1,100), padding='SAME', activation='relu'),
                         BatchNormalization(),
                         Conv2D(8,(9,1),strides=(1,100), padding='SAME', activation='relu'),
                         BatchNormalization(),  
                         
                         Conv2D(1,(129,1),strides=(1,100), padding='SAME'),

                         Flatten(),
                         # Dense(129)
                         ])

CNN_model.compile(optimizer=tf.keras.optimizers.Adam(),loss='mse',metrics=['mae'])

CNN_model.summary()


#-----------------------------------------------------------------------------
#  (4) model training 
#-----------------------------------------------------------------------------
start      = time.time()
history    = CNN_model.fit(X_train,y_train,batch_size=512,epochs=20,validation_split=0.10,verbose=1)
trian_time = time.time()-start
print("Training completed in : {:0.2f}ms".format(1000*trian_time))

# display the training progress
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.grid(color='k', linestyle='-', linewidth=0.1)
plt.show()

#-----------------------------------------------------------------------------
#  (5) verification the model
#-----------------------------------------------------------------------------

evaluation_losses = CNN_model.evaluate(X_test,y_test,verbose=0)
print()
print(" ========= test dataset has mse loss : {} ".format(evaluation_losses[0]))



#-----------------------------------------------------------------------------
#  (6) check the denoised spectrogram of the test data
#-----------------------------------------------------------------------------


# load the clean and noisy signal and it's spectrogram
file_index = 500

clean_spec = load(data_path + tb.path[file_index] + '_clean_spec.npz')['arr_0']
noisy_spec = load(data_path + tb.path[file_index] + '_noisy_spec.npz')['arr_0']

# generate input (x) with dimension 129 * 8 , and output (y) with dimension 1 * 8
# segment_len = 8
segments_number = clean_spec.shape[1] - segment_len + 1

# make the prediction based on the trained neural net
denoised_spec = np.zeros(noisy_spec.shape)

for segment_index in np.arange(segment_len - 1, clean_spec.shape[1]):

    # extract the vector elements for the data
    x_element = noisy_spec[:, segment_index - segment_len + 1: segment_index + 1]

    # scale / normalize the input sample
    x_element = (x_element - X_train_mean) / X_train_std

    # make the prediction
    y_element = CNN_model.predict(np.expand_dims(np.expand_dims(x_element, axis=0),axis=-1))

    # rescale the output back to original level
    y_element = y_element * X_train_std + X_train_mean

    # output to the spectrogram
    denoised_spec[:,segment_index] = np.reshape(y_element,(denoised_spec.shape[0]))


# plot the denoised spectrogram

plt.figure()
plt.subplot(3,1,1)
librosa.display.specshow(noisy_spec,sr=sample_rate_signal,hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency (logrithm)")
plt.title('Spectrogram of noisy audio')
plt.colorbar()

plt.subplot(3,1,2)
librosa.display.specshow(clean_spec,sr=sample_rate_signal,hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency (logrithm)")
plt.title('Spectrogram of clean audio')
plt.colorbar()

plt.subplot(3,1,3)
librosa.display.specshow(denoised_spec,sr=sample_rate_signal,hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency (logrithm)")
plt.title('Spectrogram of denoised audio')
plt.colorbar()
plt.show()

# plot the denoised spectrogram
plt.figure()
plt.subplot(3,1,1)
librosa.display.specshow(librosa.amplitude_to_db(noisy_spec),sr=sample_rate_signal,hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency (logrithm)")
plt.title('Spectrogram of noisy audio (Log scale)')
plt.colorbar()

plt.subplot(3,1,2)
librosa.display.specshow(librosa.amplitude_to_db(clean_spec),sr=sample_rate_signal,hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency (logrithm)")
plt.title('Spectrogram of clean audio (Log scale)')
plt.colorbar()

plt.subplot(3,1,3)
librosa.display.specshow(librosa.amplitude_to_db(denoised_spec),sr=sample_rate_signal,hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency (logrithm)")
plt.title('Spectrogram of denoised audio (Log scale)')
plt.colorbar()



#-----------------------------------------------------------------------------
#  (7) regenerate the time series
#-----------------------------------------------------------------------------

# load the clean and noisy signal
clean_signal = load(data_path + tb.path[file_index] + '_clean_wav.npz')['arr_0']
noisy_signal = load(data_path + tb.path[file_index] + '_noisy_wav.npz')['arr_0']

# generate the phase of spectrogram from the noisy data
noisy_stft       = librosa.core.stft(noisy_signal,hop_length=hop_length,n_fft=n_fft)
phase_noisy_stft = np.angle(noisy_stft)
mag_denoisy_stft = denoised_spec

# reconstruct the denoised  signal in time domain
denoise_signal_stft   = mag_denoisy_stft * np.cos(phase_noisy_stft) + mag_denoisy_stft  * np.sin(phase_noisy_stft)* 1j
denoise_signal        = librosa.core.istft(denoise_signal_stft)


# # test on the ifft on original signal
# test_stft        = librosa.core.stft(clean_signal,hop_length=hop_length,n_fft=n_fft)
# mag_test_stft    = np.abs(test_stft)
# pha_test_stft    = np.angle(test_stft)
#
# test_stft_rec    = mag_test_stft * np.cos(pha_test_stft) + mag_test_stft  * np.sin(pha_test_stft)* 1j
# clean_signal_rec = librosa.core.istft(test_stft_rec)


# display the clean and noisy data
plt.figure()
plt.subplot(3,1,1)
librosa.display.waveplot(noisy_signal,sr=sample_rate_signal)
plt.xlabel('')
plt.ylim([-1,1])
plt.title("Synthetic noisy signal")
plt.show()

plt.subplot(3,1,2)
librosa.display.waveplot(clean_signal,sr=sample_rate_signal)
plt.ylim([-1,1])
plt.xlabel('')
plt.title("Original clean signal")

plt.subplot(3,1,3)
librosa.display.waveplot(denoise_signal,sr=sample_rate_signal)
plt.ylim([-1,1])
plt.title("Convolutional Neural Network derived denoised signal")
plt.show()

#-----------------------------------------------------------------------------
#  (8) save the output as wav files
#-----------------------------------------------------------------------------

# sf.write(data_path + 'DenoiseTest_clean_signal.wav', clean_signal, sample_rate_signal)
# sf.write(data_path + 'DenoiseTest_Noisy_signal.wav', noisy_signal, sample_rate_signal)
sf.write(data_path + 'DenoiseTest_CNN_denoised_signal.wav', denoise_signal, sample_rate_signal)