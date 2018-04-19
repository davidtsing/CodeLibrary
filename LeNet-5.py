# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 07:51:51 2018

@author: David Tsing
"""

from skimage import io,transform
import tensorflow as tf
import numpy as np
import glob
import os

#tf.reset_default_graph()

train_path = ".\\datasets\\mnist\\train\\"
test_path = ".\\datasets\\mnist\\test\\"

def readImage(path):
    label_dir = [path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    images = []
    labels = []
    for index,folder in enumerate(label_dir):
        for img in glob.glob(folder+'/*.png'):
            image = io.imread(img)
            image = transform.resize(image,(32,32,1)) 
            images.append(image)
            labels.append(index)
    return np.asarray(images,dtype=np.float32),np.asarray(labels,dtype=np.int32)

train_data,train_label = readImage(train_path)
test_data,test_label = readImage(test_path)

#shuffle train data
train_image_num = len(train_data)
train_image_index = np.arange(train_image_num)
np.random.shuffle(train_image_index)
train_data = train_data[train_image_index]
train_label = train_label[train_image_index]
#shuffle test data
test_image_num = len(test_data)
test_image_index = np.arange(test_image_num)
np.random.shuffle(test_image_index)
test_data = test_data[test_image_index]
test_label = test_label[test_image_index]

x = tf.placeholder(tf.float32,[None,32,32,1],name='x')
y_ = tf.placeholder(tf.int32,[None],name='y_')

def inference(input_tensor,train,regularizer):
    #Layer1 - Convolution
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable('weight',[5,5,1,6],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable('bias',[6],initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='VALID')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))

    #Layer2 - Pooling
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    #Layer3 - Convolution
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable('weight',[5,5,6,16],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable('bias',[16],initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1,conv2_weights,strides=[1,1,1,1],padding='VALID')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))

    #Layer4 - Pooling
    with tf.variable_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    #Pooling to Full Connection(Parameter Reshape)
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped = tf.reshape(pool2,[-1,nodes])

    #Layer5 Size reduced from 400 to 120
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable('weight',[nodes,120],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc1_weights))
        fc1_biases = tf.get_variable('bias',[120],initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1,0.5)

    #Layer6 Size reduced from 120 to 84
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable('weight',[120,84],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc2_weights))
        fc2_biases = tf.get_variable('bias',[84],initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1,fc2_weights) + fc2_biases)
        if train:
            fc2 = tf.nn.dropout(fc2,0.5)

    #Layer7 Size reduced from 84 to 10
    with tf.variable_scope('layer7-fc3'):
        fc3_weights = tf.get_variable('weight',[84,10],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc3_weights))
        fc3_biases = tf.get_variable('bias',[10],initializer=tf.truncated_normal_initializer(stddev=0.1))
        logit = tf.matmul(fc2,fc3_weights) + fc3_biases
    return logit

y = inference(x,False,tf.contrib.layers.l2_regularizer(0.001))
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=y_)) + tf.add_n(tf.get_collection('losses'))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(y,1),tf.int32),y_),tf.float32))

def next_batch(data,label,batch_size):
    for start_index in range(0,len(data)-batch_size+1,batch_size):
        slice_index = slice(start_index,start_index+batch_size)
        yield data[slice_index],label[slice_index]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    train_num = 10
    batch_size = 64

    for i in range(train_num):
        print("train num:",i)
        train_cost,train_acc,batch_num = 0, 0, 0
        for train_data_batch,train_label_batch in next_batch(train_data,train_label,batch_size):
            _,err,acc = sess.run([optimizer,cost,accuracy],feed_dict={x:train_data_batch,y_:train_label_batch})
            train_cost+=err;train_acc+=acc;batch_num+=1
        print("train cost:",train_cost/batch_num)
        print("train acc:",train_acc/batch_num)

        test_cost,test_acc,batch_num = 0, 0, 0
        for test_data_batch,test_label_batch in next_batch(test_data,test_label,batch_size):
            err,acc = sess.run([cost,accuracy],feed_dict={x:test_data_batch,y_:test_label_batch})
            test_cost+=err;test_acc+=acc;batch_num+=1
        print("test cost:",test_cost/batch_num)
        print("test acc:",test_acc/batch_num)














