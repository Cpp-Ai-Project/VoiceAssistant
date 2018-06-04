import tensorflow as tf
from tensorflow.contrib import rnn

#MNIST data set
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#28 individual time steps
time_steps = 28
#hidden LSTM units
num_units = 128
#28 x (28-pixel row) input
n_input = 28
#learning rate for AdamOptimizer
learning_rate = 0.001
#0-9 -> 10 "classes" or possible outputs
n_classes = 10
#size of data in one iteration
batch_size = 100

#weights and biases of appropriate shape to accomplish above task
out_weights = tf.Variable(tf.random_normal([num_units,n_classes]))
out_bias = tf.Variable(tf.random_normal([n_classes]))

#initialize input (x) and output (y) (image and label, respectively)
x = tf.placeholder("float",[None,time_steps,n_input])
y = tf.placeholder("float",[None,n_classes])

#process the input tensor from [batch_size,n_steps,n_input] to "time_steps" number of [batch_size,n_input] tensors
input = tf.unstack(x ,time_steps,1)

#defining the network
lstm_layer = rnn.BasicLSTMCell(num_units,forget_bias=1)
outputs,_ = rnn.static_rnn(lstm_layer,input,dtype="float32")

#train network
def train_neural_network(x):
	#run last output through recurrent network to find next prediction, i.e. (output)(weight) + bias
	prediction = tf.matmul(outputs[-1],out_weights)+out_bias

	#calculate cost w/softmax (used to optimize model)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
	#tf library optimizes data:
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

	#model evaluation
	correct=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
	accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))

	#initialize variables
	with tf.Session() as sess:
	    sess.run(tf.global_variables_initializer())

	    #number of passes of entire data through deep net
	    n_epochs = 500
	    iter=1

	    while iter < n_epochs:
	        batch_x,batch_y = mnist.train.next_batch(batch_size=batch_size)

	        batch_x = batch_x.reshape((batch_size,time_steps,n_input))

	        sess.run(optimizer, feed_dict = {x: batch_x, y: batch_y})

	        if iter %10==0:
	            acc = sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
	            loss = sess.run(cost,feed_dict={x:batch_x,y:batch_y})
	            print("For iteration ",iter)
	            print("Accuracy: ",acc)
	            print("Loss: ",loss)
	            print("__________________")

	        iter=iter+1

train_neural_network(x)	