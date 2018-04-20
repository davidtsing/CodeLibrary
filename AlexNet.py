# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 21:04:21 2018

@author: David Tsing
"""

import tensorflow as tf
import cifar10_input
#import cifar10
import numpy as np
import math

#cifar10.maybe_download_and_extract()

tf.reset_default_graph()

data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'

def conv2d(_x, _w, _b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(_x, _w, [1, 1, 1, 1], padding='SAME'), _b))

def lrn(_x):
    return tf.nn.lrn(_x, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

def max_pool(_x, f):
    return tf.nn.max_pool(_x, [1, f, f, 1], [1, 1, 1, 1], padding='SAME')

def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses',cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def inference(image_holder, label_holder, dropout):
    #Layer1 - Convolution (Input–>Conv–>ReLUs–>LRN–>max-pooling)
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable('weight',[3, 3, 3, 32],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable('bias',[32],initializer=tf.constant_initializer(0.0))
        conv1 = conv2d(image_holder, conv1_weights, conv1_biases)
        lrn1 = lrn(conv1)
        pool1 = max_pool(lrn1, 2)
        
    #Layer2 - Convolution(Conv–>ReLUs–>LRN–>max-pooling)
    with tf.variable_scope('layer2-conv2'):
        conv2_weights = tf.get_variable('weight',[3, 3, 32, 64],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable('bias',[64],initializer=tf.constant_initializer(0.0))
        conv2 = conv2d(pool1, conv2_weights, conv2_biases)
        lrn2 = lrn(conv2)
        pool2 = max_pool(lrn2, 2)
        
    #Layer3 - Convolution(Conv–>ReLUs)
    with tf.variable_scope('layer3-conv3'):
        conv3_weights = tf.get_variable('weight',[3, 3, 64, 64],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable('bias',[64],initializer=tf.constant_initializer(0.0))
        conv3 = conv2d(pool2, conv3_weights, conv3_biases)
        
    #Layer4 - Convolution(Conv–>ReLUs)
    with tf.variable_scope('layer4-conv4'):
        conv4_weights = tf.get_variable('weight',[3, 3, 64, 256],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable('bias',[256],initializer=tf.constant_initializer(0.0))
        conv4 = conv2d(conv3, conv4_weights, conv4_biases)
        
    #Layer5 - Convolution(Conv–>ReLUs->pooling)
    with tf.variable_scope('layer5-conv5'):
        conv5_weights = tf.get_variable('weight',[3, 3, 256, 128],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv5_biases = tf.get_variable('bias',[128],initializer=tf.constant_initializer(0.0))
        conv5 = conv2d(conv4, conv5_weights, conv5_biases)
        pool5 = max_pool(conv5, 2)
    
    #Layer6 - Full Connection(pooling->vector->matmul–>ReLUs–>dropout)
    with tf.variable_scope('layer6-fc1'):
        fc1_weights = tf.get_variable('weight',[128*24*24, 1024],initializer=tf.truncated_normal_initializer(stddev=0.01))
        fc1_biases = tf.get_variable('bias',[1024],initializer=tf.constant_initializer(0.0))
        shape = pool5.get_shape()
        reshape = tf.reshape(pool5, [-1, shape[1].value*shape[2].value*shape[3].value])
        fc1 = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        fc1_drop = tf.nn.dropout(fc1, keep_prob=dropout)
        
    #Layer7 - Full Connection((matmul–>ReLUs–>dropout)
    with tf.variable_scope('layer7-fc2'):
        fc2_weights = tf.get_variable('weight',[1024, 1024],initializer=tf.truncated_normal_initializer(stddev=0.01))
        fc2_biases = tf.get_variable('bias',[1024],initializer=tf.constant_initializer(0.0))
        fc2 = tf.nn.relu(tf.matmul(fc1_drop, fc2_weights) + fc2_biases)
        fc2_drop = tf.nn.dropout(fc2, keep_prob=dropout)
        
    #Layer8 - Full Connection((matmul–>ReLUs–>softmax)
    with tf.variable_scope('layer8-fc3'):
        fc3_weights = tf.get_variable('weight',[1024, 10],initializer=tf.truncated_normal_initializer(stddev=0.01))
        fc3_biases = tf.get_variable('bias',[10],initializer=tf.constant_initializer(0.0))
        logits = tf.add(tf.matmul(fc2_drop, fc3_weights) , fc3_biases)
    return logits

STEPS = 10000
batch_size =128

images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)
images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)

image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])

y = inference(image_holder, label_holder, 0.7)
cost = loss(y, label_holder)
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator() 
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    for step in range(STEPS):
        batch_xs, batch_ys = sess.run([images_train, labels_train])
        _, loss_value = sess.run([optimizer, cost], feed_dict={image_holder: batch_xs,
                                                                label_holder: batch_ys} )
        if step % 20 == 0:
            print('step:%5d. --lost:%.6f. '%(step, loss_value))
    print('train over!')
        
    num_examples = 10000
    num_iter = int(math.ceil(num_examples / batch_size))
    true_count = 0
    total_sample_count = num_iter * batch_size  
    step = 0
    top_k_op = tf.nn.in_top_k(y, label_holder, 1)
    
    while step < num_iter:
        image_batch, label_batch = sess.run([images_test, labels_test])
        predictions = sess.run([top_k_op], feed_dict={image_holder: image_batch,
                                                      label_holder: label_batch})
        true_count += np.sum(predictions)
        step += 1
    
    precision = true_count / total_sample_count
    print('precision @ 1=%.3f' % precision)
    coord.request_stop()
    coord.join(threads) 
