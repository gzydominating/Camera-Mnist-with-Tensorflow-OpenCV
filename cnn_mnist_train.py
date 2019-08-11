import numpy as np
import tensorflow as tf 
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) 
x_input = tf.placeholder(tf.float32, [None, 784])  
y_actual = tf.placeholder(tf.float32, shape=[None, 10]) 
keep_prob = tf.placeholder("float")

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

#input -> conv -> pool -> conv -> pool -> fc -> dropout -> softmax
def network(x):
    x_image = tf.reshape(x, [-1,28,28,1]) #-1 means arbitrary
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)      #conv1
    h_pool1 = max_pool(h_conv1)                                   #max_pool1

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)      #conv2
    h_pool2 = max_pool(h_conv2)                                   #max_pool2

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)    #fc1

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)                  #dropout

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_predicts=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) #fc2 output
    return y_predicts

y_predict=network(x_input)

#see http://www.tensorfly.cn/tfdoc/tutorials/mnist_pros.html
cross_entropy = -tf.reduce_sum(y_actual*tf.log(y_predict))
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_actual,1))    
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) 

saver = tf.train.Saver()
sess=tf.InteractiveSession()                          
sess.run(tf.initialize_all_variables())
for i in range(20000):  #iteration 20000 steps = (epochs * train_size) / batch_size ,epochs=21
  batch = mnist.train.next_batch(64) #batch_size=64
  if i%100 == 0:
    train_acc = accuracy.eval(feed_dict={x_input:batch[0], y_actual: batch[1], keep_prob: 1.0})
    print('step',i,'training accuracy',train_acc)
    train_step.run(feed_dict={x_input: batch[0], y_actual: batch[1], keep_prob: 0.5})
  saver.save(sess, "model_save.ckpt") #save model
#test accuracy in mnist.test dataset
test_acc=accuracy.eval(feed_dict={x_input: mnist.test.images, y_actual: mnist.test.labels, keep_prob: 1.0})
print("test accuracy",test_acc)
