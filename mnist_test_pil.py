import numpy as np
import tensorflow as tf
from PIL import Image


def load_image(img_path):
    img = Image.open(img_path)
    return img

def resize_image(in_image, new_width, new_height, out_image=None,
                 resize_mode=Image.ANTIALIAS):
    img = in_image.resize((new_width, new_height), resize_mode)
    if out_image:
        img.save(out_image)
    return img

def pil_to_nparray(pil_image):
    pil_image.load()
    return np.asarray(pil_image, dtype="float32")

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

    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)                  #dropout

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_predicts=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) #fc2 output
    return y_predicts  

img_path = 'test_0.png'  #test image
img = load_image(img_path)
img = resize_image(img,28,28)
np_img = pil_to_nparray(img.convert("L")) #img.convert("L"): convert to gray-scale image
#np_img=np_img.reshape([-1, 28, 28, 1])
predict=network(np_img)
sess=tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "./model_save.ckpt") #load model file must have ./ with tensorflow1.0
predictions = sess.run(predict,feed_dict={keep_prob: 0.5})
predictlist=predictions.tolist() #tensorflow output is numpy.ndarray like [[0 0 0 0]]
label=predictlist[0]
result=label.index(max(label))
print('result num:')
print(result)

