'''
<PV-22> Make STT Model 
STT model that currently implements the LSTM RNN with CTC loss
Will eventually improve it so it will be Bidirection RNN and more layers

Used With California Polythechnic University California, Pomona Voice Assitant Project
Author: Jason Chang
Project Manager: Gerry Fernando Patia
Date: 10 June, 2018
'''

import time
import tensorflow as tf
import numpy as np
import os

#from create_featuresets import create_feature_sets_and_labels
from create_featuresets import getData
orginal_path = os.getcwd()

#Currently having problems feeding data with multiple files. 
#train_x_input, train_x_seq, train_y, test_x_input, test_x_seq, test_y = create_feature_sets_and_labels()

train_x_input, train_x_seq, train_y, test_x_input, test_x_seq, test_y, original = getData()

# Constants
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space

# Some configs
num_features = 26
# Number of units in the LSTM cell
num_units=50 
# Accounting the 0th indice +  space + blank label = 28 characters
num_classes = ord('z') - ord('a') + 1 + 1 + 1

# Hyper-parameters
num_epochs = 150
num_hidden = 50
num_layers = 1	#For now 1 layer for fast training
batch_size = 1

num_examples = 1
num_batches_per_epoch = int(num_examples/batch_size)

# Has size [batch_size, max_stepsize, num_features], but the batch_size and max_stepsize can vary along each step
x = tf.placeholder(tf.float32, [None, None, num_features])
# Here we use sparse_placeholder that will generate a SparseTensor required by ctc_loss op.
y = tf.sparse_placeholder(tf.int32)
# 1d array of size [batch_size]
seq_len = tf.placeholder(tf.int32, [None])

def recurrent_neural_network(x):
	#For creating multi layered RNN cell for later 
	cells = []
	for _ in range(num_layers):
		cell = tf.contrib.rnn.LSTMCell(num_units)
		cells.append(cell)
	stack = tf.contrib.rnn.MultiRNNCell(cells)

	#Initializing the Weights and Biases
	layer = {'weights':tf.Variable(tf.random_normal([num_hidden, num_classes])),
			 'biases':tf.Variable(tf.random_normal([num_classes]))}

	# The second output is the last state and we will no use that
	outputs, _ = tf.nn.dynamic_rnn(stack, x, seq_len, dtype=tf.float32)

	shape = tf.shape(x)
	batch_s, max_timesteps = shape[0], shape[1]

	# Reshaping to apply the same weights over the timesteps
	outputs = tf.reshape(outputs, [-1, num_hidden])
	output = tf.matmul(outputs, layer['weights']) + layer['biases']

	# Reshaping back to the original shape
	output = tf.reshape(output, [batch_s, -1, num_classes])

	# Time major
	output = tf.transpose(output, (1, 0, 2))

	return output

def train_neural_network(x):
	predition = recurrent_neural_network(x)
	loss = tf.nn.ctc_loss(y, predition, seq_len)
	cost = tf.reduce_mean(loss)
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	# Option 2: tf.nn.ctc_beam_search_decoder
	# (it's slower but you'll get better results)
	decoded, log_prob = tf.nn.ctc_greedy_decoder(predition, seq_len)

	# Inaccuracy: label error rate
	ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), y))

	#Creating Folder for RNN Model
	os.chdir(orginal_path)
	spec_path = os.path.join(orginal_path, 'RNNmodel\\')
	if not os.path.exists(spec_path):
		#create the RNNmodel folder if it doesn't already exist
		os.makedirs(spec_path)

	saver = tf.train.Saver()
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		spec_path = os.path.join(spec_path, 'RNNmodel.ckpt')

		for curr_epoch in range(num_epochs):
			train_cost = train_ler = 0
			start = time.time()
			#i = 0

			for batch in range(num_batches_per_epoch):
				'''
				#Trying with multiple file but getting wrong feed value of shape
				#ValueError: Cannot feed value of shape (1, 1, 246, 13) for Tensor 'Placeholder:0', which has shape '(?, ?, 13)'

				start = i
				end = i + batch_size
				batch_x_input = np.array(train_x_input[start:end])
				batch_x_seq = np.array(train_x_seq[start:end])
				batch_y = np.array(train_y[start:end])
				feed = {x: batch_x_input,
						y: batch_y,
						seq_len: batch_x_seq}
				'''

				#Input is only one data with format of [?, ?, 26]
				feed = {x: train_x_input,
						y: train_y,
						seq_len: train_x_seq}

				batch_cost, _ = sess.run([cost, optimizer], feed)			#sess.run([optimizer, cost], feed) does not work? works in different code
				train_cost += batch_cost * batch_size
				train_ler += sess.run(ler, feed_dict=feed) * batch_size
				#i += batch_size

			#Saves the model and currently commented cause of time
			#save_path = saver.save(sess, spec_path)
			#print("Model saved in path: %s" % save_path) 

			train_cost /= num_examples
			train_ler /= num_examples

			#May need to change when doing multiple test values?
			val_feed = {x: test_x_input,
						y: test_y,
						seq_len: test_x_seq}

			val_cost, val_ler = sess.run([cost, ler], feed_dict=val_feed)

			log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}"
			print(log.format(curr_epoch+1, num_epochs, train_cost, train_ler, val_cost, val_ler, time.time() - start))

		# Decoding
		d = sess.run(decoded[0], feed_dict=feed)
		str_decoded = ''.join([chr(x) for x in np.asarray(d[1]) + FIRST_INDEX])
		# Replacing blank label to none
		str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
		# Replacing space label to space
		str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')

		print('Original:\n%s' % original)
		print('Decoded:\n%s' % str_decoded)

train_neural_network(x)