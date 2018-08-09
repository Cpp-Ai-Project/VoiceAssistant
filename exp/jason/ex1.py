'''

This is a very basic program to demonstrate a deep feedforward
network with 3 hidden layers using TensorFlow. This simple 
neural network typically achieves 95% accuracy on the MNIST 
dataset of 28x28-pixel numbers 0-9 with 10 epochs by using 
one-hot encoding to predict the written number 0-9.

'''

import tensorflow as tf

#MNIST data set
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#number of neurons at each hidden layer
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

#0-9 -> 10 "classes" or possible outputs
n_classes = 10
#size of data in one iteration
batch_size = 100

#initialize input (x) and output (y)
x = tf.placeholder('float',[None,784])
y = tf.placeholder('float')

def neural_net_model(data):
	#set up layers for deep net: x->l1->l2->l3->out
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([28*28,n_nodes_hl1])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),
					'biases':tf.Variable(tf.random_normal([n_classes]))}

	#(inputs)(weights) + bias, at each layer:
	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']),hidden_1_layer['biases'])
	#rectify linear (as activation function)
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']),hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)
	
	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']),hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	# "+" same as "tf.add"
	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

	return output

def train_neural_network(x):
	prediction = neural_net_model(x)
	#calculate cost w/softmax (used to optimize model)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
	#tf library optimizes data:
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	#number of passes of entire data through deep net
	n_epochs = 10

	with tf.Session() as sess:
		#init variables
		sess.run(tf.global_variables_initializer())
		for epoch in range(n_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x,epoch_y  = mnist.train.next_batch(batch_size)
				# "_" used for variable which is not used besides to run session
				_,epoch_c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
				epoch_loss += epoch_c
			print('Epoch', epoch + 1, 'completed out of ', n_epochs, 'loss: ', epoch_loss)
		#compare prediction (prediction) to output (y)
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
		#average accuracy printed at end of train session
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:', accuracy.eval({x:mnist.test.images, y: mnist.test.labels}))

train_neural_network(x)	