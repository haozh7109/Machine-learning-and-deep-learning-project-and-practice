#!/usr/bin/env python
# coding: utf-8

# ## Language model for the Shakespeare dataset

# # Code by Hao, Zhao, July, 10, 2020.

# ### Motivations:
# 
# Use the text preprocessing tools and RNN models to build a character-level language model, and then train the model on the works of Shakespeare, and use the network to generate the similar style of own text..
# 

# In[1]:


#### PACKAGE IMPORTS ####

# Run this cell first to import all required packages. Do not make any imports elsewhere in the notebook

import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

# If you would like to make further imports from tensorflow, add them here
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Embedding, GRU 
from tensorflow.keras.models import Model,Sequential


# ![Shakespeare image](test_data_folder/shakespeare/data/shakespeare.png)
# 
# #### The Shakespeare dataset
# 
# In this assignment, you will use a subset of the [Shakespeare dataset](http://shakespeare.mit.edu). It consists of a single text file with several excerpts concatenated together. The data is in raw text form, and so far has not yet had any preprocessing. 
# 
# Your goal is to construct an unsupervised character-level sequence model that can generate text according to a distribution learned from the dataset.

# #### Load and inspect the dataset

# In[2]:


# Load the text file into a string

with open('test_data_folder/shakespeare/data/Shakespeare.txt', 'r', encoding='utf-8') as file:
    text = file.read()


# In[3]:


# Create a list of chunks of text

text_chunks = text.split('.')


# To give you a feel for what the text looks like, we will print a few chunks from the list.

# In[4]:


# Display some randomly selected text samples

num_samples = 5
inx = np.random.choice(len(text_chunks), num_samples, replace=False)
for chunk in np.array(text_chunks)[inx]:
    print(chunk)


# #### Create a character-level tokenizer

# You should now write a function that returns a `Tokenizer` object. The function takes a list of strings as an argument, and should create a `Tokenizer` according to the following specification:
# 
# * The number of tokens should be unlimited (there should be as many as required by the dataset).
# * Tokens should be created at the character level (not at the word level, which is the default behaviour).
# * No characters should be filtered out or ignored.
# * The original capitalization should be retained (do not convert the text to lower case)
# 
# The `Tokenizer` should be fit to the `list_of_strings` argument and returned by the function. 
# 
# **Hint:** you may need to refer to the [documentation](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer) for the `Tokenizer`.

# In[5]:


#### GRADED CELL ####

# Complete the following function.
# Make sure not to change the function name or arguments.

def create_character_tokenizer(list_of_strings):
    """
    This function takes a list of strings as its argument. It should create 
    and return a Tokenizer according to the above specifications. 
    """
    tokenizer = Tokenizer(num_words=None,
                         char_level=True,
                         filters=None,
                         lower=False,
                         )
    
    tokenizer.fit_on_texts(list_of_strings)
    
    return tokenizer


# In[6]:


# Get the tokenizer

tokenizer = create_character_tokenizer(text_chunks)


# #### Tokenize the text
# 
# You should now write a function to use the tokenizer to map each string in `text_chunks` to its corresponding encoded sequence. The following function takes a fitted `Tokenizer` object in the first argument (as returned by `create_character_tokenizer`) and a list of strings in the second argument. The function should return a list of lists, where each sublist is a sequence of integer tokens encoding the text sequences according to the mapping stored in the tokenizer.
# 
# **Hint:** you may need to refer to the [documentation](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer) for the `Tokenizer`.

# In[7]:


#### GRADED CELL ####

# Complete the following function.
# Make sure not to change the function name or arguments.

def strings_to_sequences(tokenizer, list_of_strings):
    """
    This function takes a tokenizer object and a list of strings as its arguments.
    It should use the tokenizer to map the text chunks to sequences of tokens and
    then return this list of encoded sequences.
    """
    
    sequences_encoded = tokenizer.texts_to_sequences(list_of_strings)
    
    return sequences_encoded
    


# In[8]:


# Encode the text chunks into tokens

seq_chunks = strings_to_sequences(tokenizer, text_chunks)


# #### Pad the encoded sequences and store them in a numpy array

# Since not all of the text chunks are the same length, you will need to pad them in order to train on batches. You should now complete the following function, which takes the list of lists of tokens, and creates a single numpy array with the token sequences in the rows, according to the following specification:
# 
# * The longest allowed sequence should be 500 tokens. Any sequence that is longer should be shortened by truncating the beginning of the sequence.
# * Use zeros for padding the sequences. The zero padding should be placed before the sequences as required.
# 
# The function should then return the resulting numpy array.
# 
# **Hint:** you may want to refer to the [documentation](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences) for the `pad_sequences` function.

# In[9]:


#### GRADED CELL ####

# Complete the following function.
# Make sure not to change the function name or arguments.

def make_padded_dataset(sequence_chunks):
    """
    This function takes a list of lists of tokenized sequences, and transforms
    them into a 2D numpy array, padding the sequences as necessary according to
    the above specification. The function should then return the numpy array.
    """
    sequences_padded = tf.keras.preprocessing.sequence.pad_sequences(sequence_chunks,maxlen=500,padding='pre',value=0,truncating='pre')
    
    return sequences_padded
    
    


# In[10]:


# Pad the token sequence chunks and get the numpy array

padded_sequences = make_padded_dataset(seq_chunks)


# #### Create model inputs and targets
# 
# Now you are ready to build your RNN model. The model will receive a sequence of characters and predict the next character in the sequence. At training time, the model can be passed an input sequence, with the target sequence is shifted by one.
# 
# For example, the expression `To be or not to be` appears in Shakespeare's play 'Hamlet'. Given input `To be or not to b`, the correct prediction is `o be or not to be`. Notice that the prediction is the same length as the input!
# 
# ![sequence_prediction_example](test_data_folder/shakespeare/data/rnn_example.png)
# 
# You should now write the following function to create an input and target array from the current `padded_sequences` array. The function has a single argument that is a 2D numpy array of shape `(num_examples, max_seq_len)`. It should fulfil the following specification:
# 
# * The function should return an input array and an output array, both of size `(num_examples, max_seq_len - 1)`.
# * The input array should contain the first `max_seq_len - 1` tokens of each sequence. 
# * The output array should contain the last `max_seq_len - 1` tokens of each sequence. 
# 
# The function should then return the tuple `(input_array, output_array)`. Note that it is possible to complete this function using numpy indexing alone!

# In[11]:


#### GRADED CELL ####

# Complete the following function.
# Make sure not to change the function name or arguments.

def create_inputs_and_targets(array_of_sequences):
    """
    This function takes a 2D numpy array of token sequences, and returns a tuple of two
    elements: the first element is the input array and the second element is the output
    array, which are defined according to the above specification.
    """   
    max_seq_len  = array_of_sequences.shape[1]
    input_array  = array_of_sequences [:, : (max_seq_len - 1)]
    output_array = array_of_sequences [:, -(max_seq_len-1):]
    return input_array,output_array


# In[12]:


# Create the input and output arrays

input_seq, target_seq = create_inputs_and_targets(padded_sequences)


# #### Preprocess sequence array for stateful RNN
# 
# We will build our RNN language model to be stateful, so that the internal state of the RNN will be maintained across batches. For this to be effective, we need to make sure that each element of every batch follows on from the corresponding element of the preceding batch (you may want to look back at the "Stateful RNNs" reading notebook earlier in the week).
# 
# The following code processes the input and output sequence arrays so that they are ready to be split into batches for training a stateful RNN, by re-ordering the sequence examples (the rows) according to a specified batch size. 

# In[13]:


# Fix the batch size for training

batch_size = 32


# In[14]:


# Prepare input and output arrays for training the stateful RNN

num_examples = input_seq.shape[0]

num_processed_examples = num_examples - (num_examples % batch_size)

input_seq = input_seq[:num_processed_examples]
target_seq = target_seq[:num_processed_examples]

steps = int(num_processed_examples / 32)  # steps per epoch

inx = np.empty((0,), dtype=np.int32)
for i in range(steps):
    inx = np.concatenate((inx, i + np.arange(0, num_processed_examples, steps)))

input_seq_stateful = input_seq[inx]
target_seq_stateful = target_seq[inx]


# #### Split the data into training and validation sets
# 
# We will set aside approximately 20% of the data for validation.

# In[15]:


# Create the training and validation splits

num_train_examples = int(batch_size * ((0.8 * num_processed_examples) // batch_size))

input_train = input_seq_stateful[:num_train_examples]
target_train = target_seq_stateful[:num_train_examples]

input_valid = input_seq_stateful[num_train_examples:]
target_valid = target_seq_stateful[num_train_examples:]


# #### Create training and validation Dataset objects
# 
# You should now write a function to take the training and validation input and target arrays, and create training and validation `tf.data.Dataset` objects. The function takes an input array and target array in the first two arguments, and the batch size in the third argument. Your function should do the following:
# 
# * Create a `Dataset` using the `from_tensor_slices` static method, passing in a tuple of the input and output numpy arrays.
# * Batch the `Dataset` using the `batch_size` argument, setting `drop_remainder` to `True`. 
# 
# The function should then return the `Dataset` object.

# In[16]:


#### GRADED CELL ####

# Complete the following function.
# Make sure not to change the function name or arguments.

def make_Dataset(input_array, target_array, batch_size):
    """
    This function takes two 2D numpy arrays in the first two arguments, and an integer
    batch_size in the third argument. It should create and return a Dataset object 
    using the two numpy arrays and batch size according to the above specification.
    """
    dataset = tf.data.Dataset.from_tensor_slices((input_array,target_array))
    dataset = dataset.batch(batch_size,drop_remainder=True)
    
    return dataset
    


# In[17]:


# Create the training and validation Datasets

train_data = make_Dataset(input_train, target_train, batch_size)
valid_data = make_Dataset(input_valid, target_valid, batch_size)


# #### Build the recurrent neural network model

# You are now ready to build your RNN character-level language model. You should write the following function to build the model; the function takes arguments for the batch size and vocabulary size (number of tokens). Using the Sequential API, your function should build your model according to the following specifications:
# 
# * The first layer should be an Embedding layer with an embedding dimension of 256 and set the vocabulary size to `vocab_size` from the function argument.
# * The Embedding layer should also mask the zero padding in the input sequences.
# * The Embedding layer should also set the `batch_input_shape` to `(batch_size, None)` (a fixed batch size is required for stateful RNNs).
# * The next layer should be a (uni-directional) GRU layer with 1024 units, set to be a stateful RNN layer.
# * The GRU layer should return the full sequence, instead of just the output state at the final time step.
# * The final layer should be a Dense layer with `vocab_size` units and no activation function.
# 
# In total, the network should have 3 layers.

# In[18]:


#### GRADED CELL ####

# Complete the following function.
# Make sure not to change the function name or arguments.

def get_model(vocab_size, batch_size):
    """
    This function takes a vocabulary size and batch size, and builds and returns a 
    Sequential model according to the above specification.
    """
    model = Sequential([Embedding(input_dim=vocab_size, output_dim=256, mask_zero=True, batch_input_shape = (batch_size,None)),
                        GRU(1024,stateful=True,return_sequences=True),
                        Dense(vocab_size)
                       ])
    return model


# In[19]:


# Build the model and print the model summary

model = get_model(len(tokenizer.word_index) + 1, batch_size)
model.summary()


# #### Compile and train the model
# 
# You are now ready to compile and train the model. For this model and dataset, the training time is very long. Therefore for this assignment it is not a requirement to train the model. We have pre-trained a model for you (using the code below) and saved the model weights, which can be loaded to get the model predictions. 
# 
# It is recommended to use accelerator hardware (e.g. using Colab) when training this model. It would also be beneficial to increase the size of the model, e.g. by stacking extra recurrent layers.

# In[20]:


# Choose whether to train a new model or load the pre-trained model

skip_training = True


# In[21]:


# Compile and train the model, or load pre-trained weights

if not skip_training:
    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(filepath='./test_data_folder/shakespeare/models/ckpt',
                                                           save_weights_only=True,
                                                           save_best_only=True)
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['sparse_categorical_accuracy'])
    history = model.fit(train_data, epochs=15, validation_data=valid_data, 
                        validation_steps=50, callbacks=[checkpoint_callback])


# In[22]:


# Save model history as a json file, or load it if using pre-trained weights

if not skip_training:
    history_dict = dict()
    for k, v in history.history.items():
        history_dict[k] = [float(val) for val in history.history[k]]
    with open('test_data_folder/shakespeare/models/history.json', 'w+') as json_file:
        json.dump(history_dict, json_file, sort_keys=True, indent=4)
else:
    with open('test_data_folder/shakespeare/models/history.json', 'r') as json_file:
        history_dict = json.load(json_file)


# #### Plot the learning curves

# In[23]:


# Run this cell to plot accuracy vs epoch and loss vs epoch

plt.figure(figsize=(15,5))
plt.subplot(121)
plt.plot(history_dict['sparse_categorical_accuracy'])
plt.plot(history_dict['val_sparse_categorical_accuracy'])
plt.title('Accuracy vs. epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.xticks(np.arange(len(history_dict['sparse_categorical_accuracy'])))
ax = plt.gca()
ax.set_xticklabels(1 + np.arange(len(history_dict['sparse_categorical_accuracy'])))
plt.legend(['Training', 'Validation'], loc='lower right')

plt.subplot(122)
plt.plot(history_dict['loss'])
plt.plot(history_dict['val_loss'])
plt.title('Loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.xticks(np.arange(len(history_dict['sparse_categorical_accuracy'])))
ax = plt.gca()
ax.set_xticklabels(1 + np.arange(len(history_dict['sparse_categorical_accuracy'])))
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show() 


# #### Write a text generation algorithm
# 
# You can now use the model to generate text! In order to generate a single text sequence, the model needs to be rebuilt with a batch size of 1.

# In[24]:


# Re-build the model and load the saved weights

model = get_model(len(tokenizer.word_index) + 1, batch_size=1)
model.load_weights(tf.train.latest_checkpoint('./test_data_folder/shakespeare/models/'))


# An algorithm to generate text is as follows:
# 
# 1. Specify a seed string (e.g. `'ROMEO:'`) to get the network started, and a define number of characters for the model to generate, `num_generation_steps`.
# 2. Tokenize this sentence to obtain a list containing one list of the integer tokens.
# 3. Reset the initial state of the network. 
# 4. Convert the token list into a Tensor (or numpy array) and pass it to your model as a batch of size one.
# 5. Get the model prediction (logits) for the last time step and extract the state of the recurrent layer.
# 6. Use the logits to construct a categorical distribution and sample a token from it.
# 7. Repeat the following for `num_generation_steps - 1` steps:
# 
#     1. Use the saved state of the recurrent layer and the last sampled token to get new logit predictions
#     2. Use the logits to construct a new categorical distribution and sample a token from it.
#     3. Save the updated state of the recurrent layer.    
# 
# 8. Take the final list of tokens and convert to text using the Tokenizer.
# 
# Note that the internal state of the recurrent layer can be accessed using the `states` property. For the GRU layer, it is a list of one variable:

# In[25]:


# Inspect the model's current recurrent state

model.layers[1].states


# We will break the algorithm down into two steps. First, you should now complete the following function that takes a sequence of tokens of any length and returns the model's prediction (the logits) for the last time step. The specification is as follows:
# 
# * The token sequence will be a python list, containing one list of integer tokens, e.g. `[[1, 2, 3, 4]]`
# * The function should convert the list into a 2D Tensor or numpy array
# * If the function argument `initial_state` is `None`, then the function should reset the state of the recurrent layer to zeros.
# * Otherwise, if the function argument `initial_state` is a 2D Tensor or numpy array, assign the value of the internal state of the GRU layer to this argument.
# * Get the model's prediction (logits) for the last time step only.
# 
# The function should then return the logits as a 2D numpy array, where the first dimension is equal to 1 (batch size).
# 
# **Hint:** the internal state of the recurrent can be reset to zeros using the `reset_states` method.

# In[26]:


#### GRADED CELL ####

# Complete the following function.
# Make sure not to change the function name or arguments.

def get_logits(model, token_sequence, initial_state=None):
    """
    This function takes a model object, a token sequence and an optional initial
    state for the recurrent layer. The function should return the logits prediction
    for the final time step as a 2D numpy array.
    """
    token_sequence_input = tf.convert_to_tensor(token_sequence, dtype=tf.float32)
    
    if initial_state is None:
        model.layers[1].reset_states()
    else:
        model.layers[1].reset_states(initial_state)
        
    predict = model.predict(token_sequence_input)
    
    output = predict[:,-1,:]
    
    return output


# In[27]:


# Test the get_logits function by passing a dummy token sequence

dummy_initial_state = tf.random.normal(model.layers[1].states[0].shape)
get_logits(model, [[1,2,3,4]], initial_state=dummy_initial_state)


# You should now write a function that takes a logits prediction similar to the above, uses it to create a categorical distribution, and samples a token from this distribution. The following function takes a 2D numpy array `logits` as an argument, and should return a single integer prediction that is sampled from the categorical distribution. 
# 
# **Hint:** you might find the `tf.random.categorical` function useful for this; see the documentation [here](https://www.tensorflow.org/api_docs/python/tf/random/categorical).

# In[28]:


#### GRADED CELL ####

# Complete the following function.
# Make sure not to change the function name or arguments.

def sample_token(logits):
    """
    This function takes a 2D numpy array as an input, and constructs a 
    categorical distribution using it. It should then sample from this
    distribution and return the sample as a single integer.
    """
    sample = tf.random.categorical(logits, num_samples=1)[0,0].numpy()
    return sample
    


# In[29]:


# Test the sample_token function by passing dummy logits

dummy_initial_state = tf.random.normal(model.layers[1].states[0].shape)
dummy_logits = get_logits(model, [[1, 2, 3, 4]], initial_state=dummy_initial_state)
sample_token(dummy_logits)


# In[30]:


logits_size = dummy_logits.shape[1]
dummy_logits = -np.inf*np.ones((1, logits_size))
dummy_logits[0, 20] = 0
sample_token(dummy_logits)
random_inx = np.random.choice(logits_size, 2, replace=False)
random_inx1, random_inx2 = random_inx[0], random_inx[1]
print(random_inx1, random_inx2)
dummy_logits = -np.inf*np.ones((1, logits_size))
dummy_logits[0, random_inx1] = 0
dummy_logits[0, random_inx2] = 0
sampled_token = []
for _ in range(100):
    sampled_token.append(sample_token(dummy_logits))
    
l_tokens, l_counts = np.unique(np.array(sampled_token), return_counts=True)
len(l_tokens) == 2


# #### Generate text from the model
# 
# You are now ready to generate text from the model!

# In[31]:


# Create a seed string and number of generation steps

init_string = 'ROMEO:'
num_generation_steps = 1000


# In[32]:


# Use the model to generate a token sequence

token_sequence = tokenizer.texts_to_sequences([init_string])
initial_state = None
input_sequence = token_sequence

for _ in range(num_generation_steps):
    logits = get_logits(model, input_sequence, initial_state=initial_state)
    sampled_token = sample_token(logits)
    token_sequence[0].append(sampled_token)
    input_sequence = [[sampled_token]]
    initial_state = model.layers[1].states[0].numpy()

print(tokenizer.sequences_to_texts(token_sequence)[0][::2])


# Congratulations for completing this programming assignment! In the next week of the course we will see how to build customised models and layers, and make custom training loops.

# In[ ]:




